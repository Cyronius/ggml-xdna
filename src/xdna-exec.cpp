#include "xdna-exec.h"

#include "xdna-env.h"
#include "xdna-npu.h"
#include "xdna-ref.h"
#include "ggml-impl.h"
#include "thread_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

int32_t param_i32(const ggml_tensor * t, int i) { return ((const int32_t *) t->op_params)[i]; }
float param_f32(const ggml_tensor * t, int i) {
    float v;
    memcpy(&v, (const int32_t *) t->op_params + i, sizeof(v));
    return v;
}

// Rows are prompt tokens, along a tensor's last dimension in use: [E, T] or
// [head dim, heads, T].
int64_t tokens_of(const ggml_tensor * t) {
    return t->ne[3] > 1 ? t->ne[3] : t->ne[2] > 1 ? t->ne[2] : t->ne[1];
}
int64_t row_floats(const ggml_tensor * t) { return ggml_nelements(t) / tokens_of(t); }

// A reshape shares its source's memory; everything produced in a piece is
// stored under the tensor that owns the memory.
const ggml_tensor * owner(const ggml_tensor * t) {
    while (t->view_src) t = t->view_src;
    return t;
}

bool is_noop(const ggml_tensor * n) {
    return n->op == GGML_OP_NONE || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW || n->op == GGML_OP_PERMUTE ||
           n->op == GGML_OP_TRANSPOSE;
}

bool f32_contig(const ggml_tensor * t) { return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t); }

// a single row as wide as a's rows (a norm's scale, a bias), broadcast over every row of a
bool is_row_of(const ggml_tensor * r, const ggml_tensor * a) {
    return f32_contig(r) && r->ne[0] == a->ne[0] && r->ne[1] == 1 && r->ne[2] == 1 && r->ne[3] == 1;
}

// The activations ggml-cpu uses, in float: its GELUs read a 16-bit table, the GPU computes them as here.
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float gelu(float x) { return 0.5f * x * (1.0f + std::tanh(0.7978845608f * x * (1.0f + 0.044715f * x * x))); }
inline float gelu_erf(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.7071067812f)); }
inline float gelu_quick(float x) { return x / (1.0f + std::exp(-1.702f * x)); }

bool glu_kind(int g) {
    return g == GGML_GLU_OP_SWIGLU || g == GGML_GLU_OP_GEGLU || g == GGML_GLU_OP_GEGLU_ERF ||
           g == GGML_GLU_OP_GEGLU_QUICK || g == GGML_GLU_OP_REGLU;
}

// ggml's YaRN rotary for one pair of dimensions (rope_yarn in ggml-cpu); the pair's index is i0 / 2.
void rope_yarn(float theta_extrap, float freq_scale, const float corr[2], int64_t pair, float ext_factor, float mscale,
               float * c, float * s) {
    float theta = freq_scale * theta_extrap;
    if (ext_factor != 0.0f) {
        const float y = ((float) pair - corr[0]) / std::max(0.001f, corr[1] - corr[0]);
        const float ramp_mix = (1 - std::min(1.0f, std::max(0.0f, y))) * ext_factor;
        theta = theta * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
    }
    *c = std::cos(theta) * mscale;
    *s = std::sin(theta) * mscale;
}

} // namespace

double xdna_pass_ms = 0;
int64_t xdna_passes = 0;

const ggml_tensor * xdna_exec_param(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL: return op->src[1];
        case GGML_OP_ADD: return op->src[1] && op->src[0] && is_row_of(op->src[1], op->src[0]) &&
                                         !ggml_are_same_shape(op->src[0], op->src[1])
                                     ? op->src[1]
                                     : nullptr;
        case GGML_OP_ROPE: return op->src[2];
        default: return nullptr;
    }
}

bool xdna_exec_supports(const ggml_tensor * op) {
    const ggml_tensor * a = op->src[0], * b = op->src[1];
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return true;  // the policy checks the weight side
        case GGML_OP_RESHAPE:
            return a && ggml_is_contiguous(a);
        case GGML_OP_RMS_NORM:
            return f32_contig(a) && f32_contig(op);
        case GGML_OP_MUL:
            // a row of scales (a norm's weight), broadcast over every row
            return f32_contig(a) && f32_contig(op) && is_row_of(b, a) && ggml_are_same_shape(a, op);
        case GGML_OP_ADD:
            // two of the same shape, or a bias row added to every row
            return f32_contig(a) && f32_contig(b) && f32_contig(op) && ggml_are_same_shape(a, op) &&
                   (ggml_are_same_shape(a, b) || is_row_of(b, a));
        case GGML_OP_SCALE:
            return f32_contig(a) && f32_contig(op) && ggml_are_same_shape(a, op);
        case GGML_OP_GLU:
            if (!glu_kind(param_i32(op, 0)) || !f32_contig(a) || !f32_contig(op)) return false;
            if (b) return param_i32(op, 1) == 0 && f32_contig(b) && ggml_are_same_shape(a, b) && ggml_are_same_shape(a, op);
            // one tensor: gate and up are its two halves
            return a->ne[0] == 2 * op->ne[0] && ggml_nrows(a) == ggml_nrows(op);
        case GGML_OP_ROPE: {
            if (!f32_contig(a) || !f32_contig(op) || !b || b->type != GGML_TYPE_I32) return false;
            const int n_dims = param_i32(op, 1), mode = param_i32(op, 2), n_offs = param_i32(op, 15);
            const ggml_tensor * ff = op->src[2];  // per-frequency factors
            if (ff && (!f32_contig(ff) || ff->ne[0] < n_dims / 2)) return false;
            return (mode == GGML_ROPE_TYPE_NEOX || mode == GGML_ROPE_TYPE_NORMAL) && n_dims > 0 && n_dims % 2 == 0 &&
                   n_dims <= a->ne[0] && n_offs == 0 && a->ne[3] == 1 && b->ne[0] == a->ne[2];
        }
        default:
            return false;
    }
}

namespace {

// Rows a pass handles at a time. Everything a tile touches (the NPU's output
// rows, each small op's result, the next input) stays in cache between steps.
constexpr int64_t TILE_ROWS = 8;

// Matmuls that share one input, run as one NPU submission (q, k and v; gate
// and up), and where each call's result goes (a fused gate/up's: its SwiGLU).
struct group {
    const ggml_tensor * x = nullptr;
    std::vector<xdna_npu::wkey> keys;
    std::vector<const ggml_tensor *> outs;
    std::vector<const ggml_tensor *> mms;  // the matmul nodes (on the host, outs are these)
};

// One pass over a stream's rows: read the results of the group that just finished, run the small ops that
// follow it, and encode the next group's input straight into the NPU buffer.
// Separate passes each went through memory, and memory bandwidth is what the
// NPU competes for.
struct segment {
    int done = -1, next = -1;  // groups, or -1
    std::vector<const ggml_tensor *> ops;
};

// Rows [r0, r1) and the calling thread's scratch.
struct tile {
    int64_t r0, r1;
    float * scratch;
};

struct piece {
    const ggml_cgraph * g;
    xdna_npu * npu;  // null: everything on the host
    const xdna_io & io;
    thread_pool & pool;
    std::vector<ggml_tensor *> nodes;                        // compute nodes and reshapes, in order
    std::unordered_set<const ggml_tensor *> produced;        // owners of memory produced here
    std::unordered_map<const ggml_tensor *, void *> mem;     // a whole tensor in host memory, by key
    // A tensor made and read within one pass lives only in tiles: its offset
    // in each thread's scratch, by key.
    std::unordered_map<const ggml_tensor *, size_t> local;
    size_t scratch_floats = 0;
    int64_t T = 0;

    piece(const ggml_cgraph * g_, xdna_npu * n, thread_pool & p, const xdna_io & i) : g(g_), npu(n), io(i), pool(p) {}

    // what a tensor's rows are kept under: its memory's owner if that's
    // produced here, else the tensor itself (an input from outside)
    const ggml_tensor * key(const ggml_tensor * t) const {
        const ggml_tensor * o = owner(t);
        return produced.count(o) ? o : t;
    }

    void * data(const ggml_tensor * t) const {
        auto it = mem.find(key(t));
        return it == mem.end() ? nullptr : it->second;
    }

    // row r of t, in the tile's scratch or in host memory
    float * row(const ggml_tensor * t, int64_t r, const tile & tl) const {
        const ggml_tensor * k = key(t);
        auto l = local.find(k);
        if (l != local.end()) return tl.scratch + l->second + (r - tl.r0) * row_floats(k);
        return (float *) mem.at(k) + r * row_floats(k);
    }

    // A rope's cos/sin for every token, [T][n_dims] as ggml_rope_cache_init
    // lays them out, with YaRN's ramp and per-frequency factors when the
    // model has them. Every layer's q and k use the same positions and
    // parameters, so it's built once and kept until either changes.
    const float * rope_table(const ggml_tensor * n) {
        static std::vector<float> table;
        static std::vector<int32_t> key_pos;
        static float key_par[8] = {};
        static const float * key_ff = nullptr;
        const ggml_tensor * b = n->src[1];
        const int32_t * pos = (const int32_t *) data(b);
        const int64_t T_ = b->ne[0];
        const int n_dims = param_i32(n, 1), n_ctx_orig = param_i32(n, 4);
        const float * ff = n->src[2] ? io.param(n->src[2]) : nullptr;
        // freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
        const float par[8] = { param_f32(n, 5), param_f32(n, 6), param_f32(n, 7), param_f32(n, 8),
                               param_f32(n, 9), param_f32(n, 10), (float) n_dims, (float) n_ctx_orig };
        if ((int64_t) key_pos.size() == T_ && memcmp(key_pos.data(), pos, (size_t) T_ * sizeof(int32_t)) == 0 &&
            memcmp(key_par, par, sizeof(par)) == 0 && key_ff == ff)
            return table.data();
        const float freq_base = par[0], freq_scale = par[1], ext_factor = par[2], attn_factor = par[3];
        float corr[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, par[4], par[5], corr);
        const float theta_scale = std::pow(freq_base, -2.0f / (float) n_dims);
        table.resize((size_t) (T_ * n_dims));
        pool.parallel_for(T_, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; t++) {
                float theta = (float) pos[t];
                float * cs = table.data() + t * n_dims;
                for (int p = 0; p < n_dims / 2; p++) {
                    rope_yarn(ff ? theta / ff[p] : theta, freq_scale, corr, p, ext_factor, attn_factor, &cs[2 * p],
                              &cs[2 * p + 1]);
                    theta *= theta_scale;
                }
            }
        });
        key_pos.assign(pos, pos + T_);
        memcpy(key_par, par, sizeof(par));
        key_ff = ff;
        return table.data();
    }

    // What a small op needs besides its inputs, fetched before any worker
    // runs it (the fetches aren't thread-safe): a MUL's scale row, a bias
    // row, a rope's cos/sin table.
    std::unordered_map<const ggml_tensor *, const float *> prep;
    void prepare(const ggml_tensor * n) {
        if (prep.count(n)) return;
        if (n->op == GGML_OP_ROPE) prep[n] = rope_table(n);
        else if (const ggml_tensor * p = xdna_exec_param(n)) prep[n] = io.param(p);
        else prep[n] = nullptr;
    }

    // A small op over the tile's rows, on the calling thread (prepare first).
    void run_rows(const ggml_tensor * n, const tile & tl) const {
        const int64_t r0 = tl.r0, r1 = tl.r1, count = (r1 - r0) * row_floats(n);
        float * out = row(n, r0, tl);
        const ggml_tensor * a = n->src[0], * b = n->src[1];
        switch (n->op) {
            case GGML_OP_RMS_NORM: {
                const float eps = param_f32(n, 0);
                const int64_t d = a->ne[0];
                const float * in = row(a, r0, tl);
                for (int64_t off = 0; off < count; off += d) {
                    const float * x = in + off;
                    float ss = 0.0f;
                    for (int64_t j = 0; j < d; j++) ss += x[j] * x[j];
                    const float sc = 1.0f / std::sqrt(ss / (float) d + eps);
                    for (int64_t j = 0; j < d; j++) out[off + j] = x[j] * sc;
                }
                break;
            }
            case GGML_OP_MUL: {
                const float * x = row(a, r0, tl), * w = prep.at(n);
                const int64_t d = a->ne[0];
                for (int64_t off = 0; off < count; off += d)
                    for (int64_t j = 0; j < d; j++) out[off + j] = x[off + j] * w[j];
                break;
            }
            case GGML_OP_ADD: {
                const float * x = row(a, r0, tl);
                if (const float * w = prep.at(n)) {  // a bias row
                    const int64_t d = a->ne[0];
                    for (int64_t off = 0; off < count; off += d)
                        for (int64_t j = 0; j < d; j++) out[off + j] = x[off + j] + w[j];
                    break;
                }
                const float * y = row(b, r0, tl);
                for (int64_t off = 0; off < count; off++) out[off] = x[off] + y[off];
                break;
            }
            case GGML_OP_SCALE: {
                const float s = param_f32(n, 0), bias = param_f32(n, 1);
                const float * x = row(a, r0, tl);
                for (int64_t off = 0; off < count; off++) out[off] = x[off] * s + bias;
                break;
            }
            case GGML_OP_GLU: {
                const int kind = param_i32(n, 0);
                const int64_t f = n->ne[0];
                const bool swapped = param_i32(n, 1) != 0;
                for (int64_t t = r0; t < r1; t++) {
                    const float * g, * u;
                    if (b) {
                        g = row(a, t, tl);
                        u = row(b, t, tl);
                    } else {
                        const float * x = row(a, t, tl);
                        g = x + (swapped ? f : 0);
                        u = x + (swapped ? 0 : f);
                    }
                    float * o = out + (t - r0) * f;
                    switch (kind) {
                        case GGML_GLU_OP_SWIGLU: for (int64_t j = 0; j < f; j++) o[j] = silu(g[j]) * u[j]; break;
                        case GGML_GLU_OP_GEGLU: for (int64_t j = 0; j < f; j++) o[j] = gelu(g[j]) * u[j]; break;
                        case GGML_GLU_OP_GEGLU_ERF: for (int64_t j = 0; j < f; j++) o[j] = gelu_erf(g[j]) * u[j]; break;
                        case GGML_GLU_OP_GEGLU_QUICK: for (int64_t j = 0; j < f; j++) o[j] = gelu_quick(g[j]) * u[j]; break;
                        default: for (int64_t j = 0; j < f; j++) o[j] = g[j] > 0.0f ? g[j] * u[j] : 0.0f; break;  // REGLU
                    }
                }
                break;
            }
            case GGML_OP_ROPE: {
                const float * x = row(a, r0, tl), * table = prep.at(n);
                const int n_dims = param_i32(n, 1), mode = param_i32(n, 2);
                const int64_t d = a->ne[0], heads = a->ne[1];
                const int64_t off_n = mode == GGML_ROPE_TYPE_NEOX ? n_dims / 2 : 1;  // where each pair's partner sits
                for (int64_t t = r0; t < r1; t++) {
                    const float * cs = table + t * n_dims;
                    for (int64_t h = 0; h < heads; h++) {
                        const float * src = x + ((t - r0) * heads + h) * d;
                        float * dst = out + ((t - r0) * heads + h) * d;
                        for (int64_t i = 0; i < n_dims; i += 2) {
                            const int64_t ic = mode == GGML_ROPE_TYPE_NEOX ? i / 2 : i;
                            const float c = cs[i], sn = cs[i + 1];
                            const float x0 = src[ic], x1 = src[ic + off_n];
                            dst[ic] = x0 * c - x1 * sn;
                            dst[ic + off_n] = x0 * sn + x1 * c;
                        }
                        for (int64_t i = n_dims; i < d; i++) dst[i] = src[i];  // past the rotated part, unchanged
                    }
                }
                break;
            }
            default:
                break;
        }
    }

    // A group's matmuls on the host, rows [t0, t1): the reference the backend
    // falls back to when the NPU fails.
    bool host_matmuls(const group & G, int64_t t0, int64_t t1, std::string & err) {
        for (const ggml_tensor * m : G.mms) {
            const void * w = io.weight_bytes(m->src[0]);
            if (!w) { err = std::string("cannot read ") + m->src[0]->name; return false; }
            ggml_tensor s0 = *m->src[0], s1 = *m->src[1], d = *m;
            s0.data = (void *) w;
            s1.data = (char *) data(m->src[1]) + t0 * s1.nb[1];
            s1.ne[1] = t1 - t0;
            d.data = (char *) data(m) + t0 * d.nb[1];
            d.ne[1] = t1 - t0;
            xdna_ref_mul_mat(&s0, &s1, &d, pool.size());
        }
        return true;
    }

    // A segment over one stream's rows [t0, t1), tile by tile. NPU rows count
    // from the stream's first. False if the NPU's results held NaN or
    // infinity; the piece then stops before anything is written back.
    bool run_pass(const segment & seg, const std::vector<group> & groups, int si, int64_t t0, int64_t t1) {
        const group * done = seg.done >= 0 ? &groups[seg.done] : nullptr;
        const group * next = seg.next >= 0 ? &groups[seg.next] : nullptr;
        const int64_t tiles = (t1 - t0 + TILE_ROWS - 1) / TILE_ROWS;
        std::atomic<bool> bad{ false };
        pool.parallel_for(tiles, [&](int64_t i0, int64_t i1) {
            thread_local std::vector<float> scratch;
            if (scratch.size() < scratch_floats) scratch.resize(scratch_floats);
            for (int64_t i = i0; i < i1; i++) {
                const tile tl = { t0 + i * TILE_ROWS, std::min(t1, t0 + (i + 1) * TILE_ROWS), scratch.data() };
                if (done && npu)
                    for (size_t k = 0; k < done->outs.size(); k++)
                        for (int64_t r = tl.r0; r < tl.r1; r++)
                            if (!npu->decode_row(si, (int) k, r - t0, row(done->outs[k], r, tl))) bad = true;
                for (const ggml_tensor * n : seg.ops) run_rows(n, tl);
                if (next && npu)
                    for (int64_t r = tl.r0; r < tl.r1; r++) npu->encode_row(si, r - t0, row(next->x, r, tl));
            }
        });
        return !bad;
    }
};

// The piece's nodes, without the leaves.
std::vector<ggml_tensor *> piece_nodes(const ggml_cgraph * g) {
    std::vector<ggml_tensor *> nodes;
    for (int i = 0; i < g->n_nodes; i++)
        if (g->nodes[i]->op != GGML_OP_NONE) nodes.push_back(g->nodes[i]);
    return nodes;
}

// SiLU(gate) * up on the NPU: a SwiGLU whose gate and up are matmuls in
// this piece on the same input, read by nothing else, runs as one fused
// matmul (output mode 2) that writes the SwiGLU's result directly. So does
// a SwiGLU on one matmul whose weight holds the gate and then the up (Phi):
// its up is null here.
struct fusion { ggml_tensor * up, * glu; };
using fusions = std::unordered_map<const ggml_tensor *, fusion>;  // by the gate matmul

fusions find_fusions(const std::vector<ggml_tensor *> & nodes, const ggml_cgraph * g, const xdna_io & io) {
    fusions fused;
    const std::unordered_set<const ggml_tensor *> in_piece(nodes.begin(), nodes.end());
    for (ggml_tensor * n : nodes) {
        if (n->op != GGML_OP_GLU || param_i32(n, 0) != GGML_GLU_OP_SWIGLU || param_i32(n, 1) != 0) continue;
        ggml_tensor * gm = n->src[0], * um = n->src[1];
        if (gm->op != GGML_OP_MUL_MAT || !in_piece.count(gm) || io.needed_outside(gm, g)) continue;
        if (um) {
            if (um->op != GGML_OP_MUL_MAT || !in_piece.count(um) || io.needed_outside(um, g)) continue;
            if (owner(gm->src[1]) != owner(um->src[1]) || gm->src[0]->type != um->src[0]->type ||
                !ggml_are_same_shape(gm->src[0], um->src[0]))
                continue;
            if (!xdna_npu::has_fused_shape(gm->src[0]->ne[0], gm->src[0]->ne[1])) continue;
        } else if (gm->src[0]->ne[1] % 2 != 0 || !xdna_npu::has_fused_shape(gm->src[0]->ne[0], gm->src[0]->ne[1] / 2)) {
            continue;
        }
        bool other = false;  // another reader in the piece
        for (ggml_tensor * m : nodes)
            for (int j = 0; j < GGML_MAX_SRC && m != n; j++)
                if (m->src[j] && (owner(m->src[j]) == gm || (um && owner(m->src[j]) == um))) other = true;
        if (other) continue;
        fused[gm] = { um, n };
    }
    return fused;
}

} // namespace

void xdna_exec_weights(const ggml_cgraph * g, const xdna_io & io,
                       std::vector<std::pair<const ggml_tensor *, const ggml_tensor *>> & out) {
    const std::vector<ggml_tensor *> nodes = piece_nodes(g);
    const fusions fused = find_fusions(nodes, g, io);
    std::unordered_set<const ggml_tensor *> fused_up;
    for (const auto & f : fused)
        if (f.second.up) fused_up.insert(f.second.up);
    for (const ggml_tensor * n : nodes) {
        if (n->op != GGML_OP_MUL_MAT || fused_up.count(n)) continue;
        auto f = fused.find(n);
        if (f == fused.end()) out.emplace_back(n->src[0], nullptr);
        else out.emplace_back(n->src[0], f->second.up ? f->second.up->src[0] : n->src[0]);
    }
}

bool xdna_exec_piece(const ggml_cgraph * g, xdna_npu * npu, thread_pool & pool, const xdna_io & io, int n_streams,
                     std::string & err) {
    piece P(g, npu, pool, io);
    P.nodes = piece_nodes(g);
    for (ggml_tensor * n : P.nodes)
        if (!is_noop(n)) P.produced.insert(n);
    if (P.nodes.empty()) return true;

    // one token count for the whole piece, or no streams
    for (ggml_tensor * n : P.nodes) {
        if (is_noop(n)) continue;
        const int64_t t = tokens_of(n);
        if (!P.T) P.T = t;
        if (t != P.T) {
            err = "a piece with mixed token counts";  // not expected from llama.cpp's graphs
            return false;
        }
    }

    fusions fused;
    if (npu) fused = find_fusions(P.nodes, g, io);
    std::unordered_set<const ggml_tensor *> fused_up;
    for (const auto & f : fused)
        if (f.second.up) fused_up.insert(f.second.up);
    std::unordered_map<const ggml_tensor *, size_t> index;
    for (size_t i = 0; i < P.nodes.size(); i++) index[P.nodes[i]] = i;

    // inputs from outside: read once. A matmul's weight is the NPU's own copy
    // and a scale or bias row or rotary factors are kept parameters, so none is read here.
    for (ggml_tensor * n : P.nodes) {
        if (n->op == GGML_OP_MUL_MAT && npu) {
            auto f = fused.find(n);
            if (f != fused.end()) {
                const ggml_tensor * up = f->second.up ? f->second.up->src[0] : n->src[0];
                if (!io.ensure_fused(n->src[0], up, err)) return false;
            } else if (!fused_up.count(n) && !io.ensure_weight(n->src[0], err)) {
                return false;
            }
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const ggml_tensor * s = n->src[j];
            if (!s || P.produced.count(owner(s)) || P.mem.count(s)) continue;
            if (n->op == GGML_OP_MUL_MAT && j == 0) continue;
            if (s == xdna_exec_param(n)) continue;
            if (n->op == GGML_OP_RESHAPE) continue;
            void * p = io.alloc(ggml_nbytes(s));
            if (!p) { err = "out of host memory"; return false; }
            io.read(s, p);
            P.mem[s] = p;
        }
    }

    // The schedule, the same for every stream: the small ops in order, and
    // each matmul, with every later one that reads the same input, as one
    // group that ends a segment. The next segment starts by reading the
    // group's results.
    auto mm_key = [](const ggml_tensor * w) {
        return xdna_npu::wkey{ w->buffer, w->data, (int) w->type, w->ne[0], w->ne[1] };
    };
    std::vector<group> groups;
    std::vector<segment> segs(1);
    {
        std::vector<char> done(P.nodes.size(), 0);
        for (size_t i = 0; i < P.nodes.size(); i++) {
            ggml_tensor * n = P.nodes[i];
            if (done[i] || is_noop(n)) continue;
            if (n->op != GGML_OP_MUL_MAT) {
                segs.back().ops.push_back(n);
                continue;
            }
            group G;
            G.x = n->src[1];
            for (size_t k = i; k < P.nodes.size(); k++) {
                ggml_tensor * m = P.nodes[k];
                if (m->op != GGML_OP_MUL_MAT || done[k] || owner(m->src[1]) != owner(G.x) || fused_up.count(m)) continue;
                done[k] = 1;
                auto f = fused.find(m);
                if (f != fused.end()) {
                    // gate, up and their SwiGLU in one call
                    const ggml_tensor * up = f->second.up ? f->second.up->src[0] : m->src[0];
                    G.keys.push_back(xdna_npu::fused_key(xdna_npu::gate_key(m->src[0], up)));
                    G.outs.push_back(f->second.glu);
                    G.mms.push_back(m);
                    if (f->second.up) done[index.at(f->second.up)] = 1;
                    done[index.at(f->second.glu)] = 1;
                } else {
                    G.keys.push_back(mm_key(m->src[0]));
                    G.outs.push_back(m);
                    G.mms.push_back(m);
                }
            }
            if (!f32_contig(G.x) || row_floats(G.x) != G.x->ne[0]) { err = "a matmul input that isn't float rows"; return false; }
            for (size_t k = 0; k < G.outs.size(); k++)
                if (npu && npu->out_width(G.keys[k]) != row_floats(G.outs[k])) { err = "a matmul output of the wrong width"; return false; }
            segs.back().next = (int) groups.size();
            segs.push_back({ (int) groups.size(), -1, {} });
            groups.push_back(std::move(G));
        }
    }

    // Where each result lives. One made and read within a single segment,
    // and read by nothing outside the piece, lives only in tiles; everything
    // else gets host memory for the whole piece.
    static const bool dump = xdna_env("GGML_XDNA_DUMP") != nullptr;  // every result kept, to show
    std::unordered_map<const ggml_tensor *, int> made, last;  // by key: the segment that makes it, the last that reads it
    auto reads = [&](const ggml_tensor * t, int si) {
        const ggml_tensor * k = P.key(t);
        if (!P.produced.count(k)) return;
        auto it = last.find(k);
        if (it == last.end() || it->second < si) last[k] = si;
    };
    for (int si = 0; si < (int) segs.size(); si++) {
        const segment & seg = segs[si];
        if (seg.done >= 0)
            for (const ggml_tensor * o : groups[seg.done].outs) made[P.key(o)] = si;
        for (const ggml_tensor * n : seg.ops) {
            made[P.key(n)] = si;
            for (int j = 0; j < GGML_MAX_SRC; j++)
                if (n->src[j]) reads(n->src[j], si);
        }
        if (seg.next >= 0) reads(groups[seg.next].x, si);
    }
    std::unordered_set<const ggml_tensor *> outside;  // keys that something outside the piece reads, through any view
    for (ggml_tensor * n : P.nodes)
        if (P.produced.count(P.key(n)) && io.needed_outside(n, g)) outside.insert(P.key(n));
    std::vector<size_t> seg_floats(segs.size(), 0);
    for (const auto & [k, si] : made) {
        auto l = last.find(k);
        // on the host a matmul writes whole results, so everything is kept
        if (npu && !dump && !outside.count(k) && (l == last.end() || l->second <= si)) {
            P.local[k] = seg_floats[si];
            seg_floats[si] += (size_t) ((TILE_ROWS * row_floats(k) + 15) / 16 * 16);  // whole cache lines
            continue;
        }
        void * p = io.alloc(ggml_nbytes(k));
        if (!p) { err = "out of host memory"; return false; }
        P.mem[k] = p;
    }
    for (size_t f : seg_floats) P.scratch_floats = std::max(P.scratch_floats, f);
    for (const segment & seg : segs)
        for (const ggml_tensor * n : seg.ops) P.prepare(n);

    // The streams: token ranges of at most 1,024 rows (the NPU's largest
    // call), whole multiples of 512 but the last. They take turns: the host
    // runs one stream's pass while the NPU runs another's matmuls.
    struct stream {
        int64_t t0, t1;
        size_t seg = 0;
        bool waiting = false;
    };
    std::vector<stream> S;
    if (!npu) n_streams = 1;  // nothing to overlap with
    const int64_t per = std::min<int64_t>(1024, std::max<int64_t>(512, ((P.T + n_streams - 1) / n_streams + 511) / 512 * 512));
    for (int64_t t0 = 0; t0 < P.T; t0 += per) S.push_back({ t0, std::min(P.T, t0 + per) });
    if (npu && !groups.empty()) {
        std::vector<std::vector<xdna_npu::wkey>> all;
        for (const group & G : groups) all.push_back(G.keys);
        for (int si = 0; si < (int) S.size(); si++)
            if (!npu->reserve(si, all, err)) return false;
    }

    size_t left = S.size();
    while (left) {
        for (int si = 0; si < (int) S.size(); si++) {
            stream & s = S[si];
            if (s.seg >= segs.size()) continue;
            if (s.waiting) {
                if (!npu->wait(si, err)) return false;
                s.waiting = false;
            }
            const segment & seg = segs[s.seg++];
            if (npu && seg.next >= 0 && !npu->begin(si, groups[seg.next].keys, s.t1 - s.t0, err)) return false;
            const auto h0 = std::chrono::steady_clock::now();
            const bool finite = P.run_pass(seg, groups, si, s.t0, s.t1);
            xdna_pass_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
            xdna_passes++;
            if (!finite) {
                err = xdna_npu::not_finite_error();
                return false;
            }
            if (seg.next >= 0 && npu) {
                if (!npu->submit(si, err)) return false;
                s.waiting = true;
            } else if (seg.next >= 0 && !P.host_matmuls(groups[seg.next], s.t0, s.t1, err)) {
                return false;
            }
            if (s.seg >= segs.size()) left--;
        }
    }

    // GGML_XDNA_DUMP=1: each small op's inputs and output, first values
    if (dump) {
        fprintf(stderr, "xdna piece:");
        for (int i = 0; i < g->n_nodes; i++) fprintf(stderr, " %s(%s)", ggml_op_name(g->nodes[i]->op), g->nodes[i]->name);
        fprintf(stderr, "\n");
        for (ggml_tensor * n : P.nodes) {
            if (is_noop(n) || n->op == GGML_OP_MUL_MAT) continue;
            auto show = [&](const char * what, const ggml_tensor * t) {
                const float * p = (const float *) P.data(t);
                fprintf(stderr, "  %s %s [%lld,%lld,%lld] %s: %.4g %.4g %.4g %.4g\n", what, t->name, (long long) t->ne[0],
                        (long long) t->ne[1], (long long) t->ne[2], P.produced.count(owner(t)) ? "here" : "read",
                        p ? p[0] : -1.0, p ? p[1] : -1.0, p ? p[2] : -1.0, p ? p[3] : -1.0);
            };
            fprintf(stderr, "xdna dump: %s %s\n", ggml_op_name(n->op), n->name);
            for (int j = 0; j < 2; j++)
                if (n->src[j] && !(n->op == GGML_OP_MUL && j == 1) && !(n->op == GGML_OP_ROPE && j == 1)) show("in ", n->src[j]);
            show("out", n);
        }
    }

    // results that something outside the piece reads
    for (ggml_tensor * n : P.nodes)
        if (!is_noop(n) && io.needed_outside(n, g)) io.write(n, P.mem.at(P.key(n)));
    return true;
}
