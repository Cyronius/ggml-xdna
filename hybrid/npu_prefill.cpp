#include "npu_prefill.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Eight exponentials at a time where MSVC's vector math library is there:
// the prefill takes hundreds of millions of them (softmax, SiLU), and one at
// a time they were a large share of the host's time.
#if defined(_MSC_VER) && defined(__AVX2__)
#define HAVE_EXP8 1
#endif

// g[i] = silu(gate[i]) * up[i]
inline void swiglu_row(const float * gate, const float * up, int64_t n, float * g) {
    int64_t i = 0;
#ifdef HAVE_EXP8
    const __m256 one = _mm256_set1_ps(1.0f), neg = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(gate + i);
        const __m256 e = _mm256_exp_ps(_mm256_xor_ps(x, neg));
        _mm256_storeu_ps(g + i, _mm256_mul_ps(_mm256_div_ps(x, _mm256_add_ps(one, e)), _mm256_loadu_ps(up + i)));
    }
#endif
    for (; i < n; i++) g[i] = silu(gate[i]) * up[i];
}

// p[i] = e^((s[i] - max s) * scale) / sum, over n values
inline void softmax_row(const float * s, int64_t n, float scale, float * p) {
    float mx = -INFINITY;
    for (int64_t i = 0; i < n; i++) mx = std::max(mx, s[i]);
    float sum = 0.0f;
    int64_t i = 0;
#ifdef HAVE_EXP8
    const __m256 vm = _mm256_set1_ps(mx), vs = _mm256_set1_ps(scale);
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        const __m256 e = _mm256_exp_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(s + i), vm), vs));
        _mm256_storeu_ps(p + i, e);
        acc = _mm256_add_ps(acc, e);
    }
    alignas(32) float part[8];
    _mm256_store_ps(part, acc);
    for (float x : part) sum += x;
#endif
    for (; i < n; i++) { p[i] = std::exp((s[i] - mx) * scale); sum += p[i]; }
    const float inv = 1.0f / sum;
    for (int64_t j = 0; j < n; j++) p[j] *= inv;
}

// y[n] = x[n] / rms(x) * w[n]
inline void rmsnorm_row(const float * x, const float * w, int64_t n, float eps, float * y) {
    float ss = 0.0f;
    for (int64_t i = 0; i < n; i++) ss += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(ss / (float) n + eps);
    for (int64_t i = 0; i < n; i++) y[i] = x[i] * scale * w[i];
}

// In place: rmsnorm over the head, then rotary in the half-split form, at
// the angles for one position (cs: cos then sin, half of each).
inline void norm_rope_head(float * v, const float * w, int64_t d, float eps, const float * cs) {
    rmsnorm_row(v, w, d, eps, v);
    const int64_t half = d / 2;
    const float * c = cs, * s = cs + half;
    for (int64_t j = 0; j < half; j++) {
        const float x0 = v[j], x1 = v[j + half];
        v[j]        = x0 * c[j] - x1 * s[j];
        v[j + half] = x0 * s[j] + x1 * c[j];
    }
}

inline float dot(const float * a, const float * b, int64_t n) {
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

const uint8_t zero_block[BFP16_BLOCK_BYTES] = {};

} // namespace

struct npu_prefill::pass {
    int64_t T = 0;
    std::vector<float> x, q, k, v, rope;  // residual; this layer's q, k, v; rotary angles [t][cos | sin]
    std::vector<float> a;                 // attention's output, when it runs on the host or the GPU
    qwen3_prefill_out * out = nullptr;
    npu_prefill_stats * st = nullptr;
    bool attn_npu = false, attn_gpu = false;
};

bool npu_prefill::make_proj(const qwen3_ref & model, int il, unsigned which, proj & p, std::string & err) {
    std::vector<float> w = model.projection_weights(il, which, p.N, p.K);
    auto dir_for = [&](int64_t M, int mode) {
        return root_ + "/" + std::to_string(M) + "x" + std::to_string(p.K) + "x" + std::to_string(p.N) + "_128x64x64_c8" +
               (mode ? "_m" + std::to_string(mode) : "");
    };
    auto built = [&](const std::string & dir) {
        FILE * f = fopen((dir + "/insts.bin").c_str(), "rb");
        if (f) fclose(f);
        return f != nullptr;
    };
    // the 16-bit mode this projection is read in, where both row counts are built
    const int want = which == qwen3_prefill_opts::GATE_UP ? 2 : 1;
    p.mode = out16 && c_tiled_ && built(dir_for(BLOCK, want)) && built(dir_for(BIG, want)) ? want : 0;
    if (p.mode == 2) {
        // gate's rows then up's, reordered so the NPU's tiles pair them
        const int64_t F = p.N / 2;
        std::vector<float> fused(w.size());
        bfp16_interleave_gate_up(w.data(), w.data() + F * p.K, F, p.K, fused.data());
        w.swap(fused);
    }
    auto key = [&](int64_t M) { return ((M * 100000 + p.K) * 100000 + p.N) * 4 + p.mode; };
    for (int64_t M : { BLOCK, BIG }) {
        if (shape_for_.count(key(M))) continue;
        const std::string dir = dir_for(M, p.mode);
        if (!built(dir)) { err = "no kernel build " + dir; return false; }
        if (shape_for_.empty() && !npu_.open(dir + "/final.xclbin", err)) return false;
        const int si = npu_.add_shape(dir + "/insts.bin", M, p.K, p.N, err);
        if (si < 0) return false;
        shape_for_[key(M)] = si;
    }
    std::vector<uint8_t> packed;
    bfp16_pack_b(w.data(), p.N, p.K, tile_, packed, pool_->size());
    p.shape = shape_for_[key(BLOCK)];
    p.shape_big = shape_for_[key(BIG)];
    p.weights = npu_.add_weights(packed, err);
    return p.weights >= 0;
}

bool npu_prefill::load(const qwen3_ref & model, const std::string & build_root, int n_threads, std::string & err) {
    model_ = &model;
    hp_ = model.hp();
    root_ = build_root;
    while (!root_.empty() && (root_.back() == '/' || root_.back() == '\\')) root_.pop_back();
    c_tiled_ = root_.size() >= 3 && root_.compare(root_.size() - 3, 3, "_ct") == 0;
    pool_ = std::make_unique<thread_pool>(n_threads);
    L_.resize(hp_.n_layer);
    for (int il = 0; il < hp_.n_layer; il++) {
        layer & L = L_[il];
        L.attn_norm = model.norm_weights(il, qwen3_ref::norm::attn);
        L.q_norm    = model.norm_weights(il, qwen3_ref::norm::q);
        L.k_norm    = model.norm_weights(il, qwen3_ref::norm::k);
        L.ffn_norm  = model.norm_weights(il, qwen3_ref::norm::ffn);
        if (!make_proj(model, il, qwen3_prefill_opts::QKV,     L.qkv,     err)) return false;
        if (!make_proj(model, il, qwen3_prefill_opts::O,       L.o,       err)) return false;
        if (!make_proj(model, il, qwen3_prefill_opts::GATE_UP, L.gate_up, err)) return false;
        if (!make_proj(model, il, qwen3_prefill_opts::DOWN,    L.down,    err)) return false;
        for (const proj * p : { &L.qkv, &L.o, &L.gate_up, &L.down }) {
            a_bytes_ = std::max(a_bytes_, (size_t) (BIG * p->K / 8) * BFP16_BLOCK_BYTES);
            c_bytes_ = std::max(c_bytes_, (size_t) (BIG * p->N) * sizeof(float));
        }
    }

    // attention's shapes, where built: scores 1024 x 384 x L, probabilities
    // times values 1024 x L x 512, for L = 512, 1024, ...
    if (hp_.n_head == 2 * hp_.n_head_kv && hp_.head_dim == 128) {
        std::string aroot = root_ + "kv";
        FILE * probe = fopen((aroot + "/1024x384x512_128x64x64_c8/insts.bin").c_str(), "rb");
        kv_once_ = probe != nullptr;
        if (probe) fclose(probe);
        if (!kv_once_) aroot = root_;
        for (int64_t Lk = BLOCK; Lk <= (kv_once_ ? KV_ROWS : 8192); Lk += BLOCK) {
            const std::string sdir = aroot + "/1024x384x" + std::to_string(Lk) + "_128x64x64_c8/insts.bin";
            const std::string pdir = aroot + "/1024x" + std::to_string(Lk) + "x512_128x64x64_c8/insts.bin";
            FILE * f1 = fopen(sdir.c_str(), "rb");
            FILE * f2 = fopen(pdir.c_str(), "rb");
            const bool have = f1 && f2;
            if (f1) fclose(f1);
            if (f2) fclose(f2);
            if (!have) break;
            const int ss = npu_.add_shape(sdir, 1024, 384, Lk, err);
            const int sp = npu_.add_shape(pdir, 1024, Lk, 512, err);
            if (ss < 0 || sp < 0) return false;
            scores_[Lk] = ss;
            pv_[Lk] = sp;
        }
        for (int g = 0; kv_once_ && g < hp_.n_head_kv; g++) {
            const int kb = npu_.add_buffer((size_t) (KV_ROWS * 3 * hp_.head_dim / 8) * BFP16_BLOCK_BYTES, err);
            const int vb = npu_.add_buffer((size_t) (512 * KV_ROWS / 8) * BFP16_BLOCK_BYTES, err);
            if (kb < 0 || vb < 0) return false;
            keys_.push_back(kb);
            vt_.push_back(vb);
        }
    }
    return true;
}

bool npu_prefill::encode_kv(pass & P, const stream & s, std::string & err) {
    const int64_t Hkv = hp_.n_head_kv, D = hp_.head_dim, Ek = Hkv * D, T = P.T, D3 = 3 * D, DB = D / 8, PVN = 512;
    const int64_t r0 = s.t0, r1 = (s.t1 + BLOCK - 1) / BLOCK * BLOCK;
    const int64_t tkb = tile_.k / 8, nk_full = KV_ROWS / tile_.k;
    const size_t group_bytes = (size_t) (BLOCK * D3 / 8) * BFP16_BLOCK_BYTES;          // 512 key rows
    const size_t tile_bytes = (size_t) (tile_.n * tile_.k / 8) * BFP16_BLOCK_BYTES;
    const clk::time_point h0 = clk::now();
    std::vector<uint8_t *> K((size_t) Hkv), V((size_t) Hkv);
    for (int64_t g = 0; g < Hkv; g++) { K[g] = npu_.buffer_map(keys_[g]); V[g] = npu_.buffer_map(vt_[g]); }
    // where v transposed's row d starts (rows past D stay zero from the start)
    std::vector<size_t> vrow((size_t) D);
    for (int64_t d = 0; d < D; d++) vrow[d] = bfp16_b_offset(d, 0, PVN, KV_ROWS, tile_);
    const float * k = P.k.data(), * v = P.v.data();
    const int64_t nrow = r1 - r0, nsb = nrow / 8;
    pool_->parallel_for(Hkv * (nrow + nsb), [&](int64_t i0, int64_t i1) {
        float lo[8], hi[8], blk[8];
        size_t off[3 * 32];
        for (int64_t i = i0; i < i1; i++) {
            if (i < Hkv * nrow) {
                // key row kt: [k | k - bfp16(k) | k], in 512-row groups
                const int64_t g = i / nrow, kt = r0 + i % nrow;
                uint8_t * dst = K[g] + (size_t) (kt / BLOCK) * group_bytes;
                bfp16_b_row(kt % BLOCK, BLOCK, D3, tile_, [&](int64_t cb, size_t o) { off[cb] = o; });
                const float * src = kt < T ? k + kt * Ek + g * D : nullptr;
                for (int64_t cb = 0; cb < DB; cb++) {
                    if (!src) {
                        for (int64_t c : { cb, cb + DB, cb + 2 * DB }) memcpy(dst + off[c], zero_block, sizeof(zero_block));
                        continue;
                    }
                    bfp16_encode_block(src + cb * 8, dst + off[cb]);
                    memcpy(dst + off[2 * DB + cb], dst + off[cb], BFP16_BLOCK_BYTES);
                    bfp16_decode_block(dst + off[cb], hi);
                    for (int j = 0; j < 8; j++) lo[j] = src[cb * 8 + j] - hi[j];
                    bfp16_encode_block(lo, dst + off[DB + cb]);
                }
            } else {
                // eight positions of v transposed, every dimension
                const int64_t j = i - Hkv * nrow, g = j / nsb, sb = r0 / 8 + j % nsb;
                uint8_t * B = V[g] + ((sb / tkb) * tile_.n * tkb + (sb % tkb) * 16) * BFP16_BLOCK_BYTES;
                const int64_t valid = std::min<int64_t>(8, T - sb * 8);
                const float * vs = valid > 0 ? v + sb * 8 * Ek + g * D : v;
                for (int64_t d = 0; d < D; d++) {
                    for (int64_t e = 0; e < 8; e++) blk[e] = e < valid ? vs[e * Ek + d] : 0.0f;
                    bfp16_encode_block(blk, B + vrow[d]);
                }
            }
        }
    });
    // to the device: s's groups of keys; s's tiles of each of v transposed's
    // first D / n columns of the array (the rest are zeros, synced at the start)
    for (int64_t g = 0; g < Hkv; g++) {
        if (!npu_.buffer_sync(keys_[g], (size_t) (r0 / BLOCK) * group_bytes, (size_t) (nrow / BLOCK) * group_bytes, err)) return false;
        for (int64_t c = 0; c < D / tile_.n; c++)
            if (!npu_.buffer_sync(vt_[g], (size_t) (c * nk_full + r0 / tile_.k) * tile_bytes, (size_t) (nrow / tile_.k) * tile_bytes, err))
                return false;
    }
    P.st->attn_in_ms += ms_since(h0);
    return true;
}

bool npu_prefill::attention_on_npu(int64_t T) const {
    return attn == attn_where::npu && scores_.count((T + BLOCK - 1) / BLOCK * BLOCK);
}

bool npu_prefill::prepare_proj(pass & P, stream & s, const proj & p, std::string & err) {
    const clk::time_point h0 = clk::now();
    std::vector<int> shapes;
    for (int j = 0; j < s.calls(); j++) shapes.push_back(s.call_m(j) == BIG ? p.shape_big : p.shape);
    if (!s.proj->prepare(shapes, err, { p.weights })) return false;
    // the last call's rows past the prompt: zeros (the buffer held other data)
    const int j = s.calls() - 1;
    const int64_t rows = s.call_rows(j), M = s.call_m(j);
    uint8_t * A = s.proj->a(j);
    pool_->parallel_for(M - rows, [&](int64_t i0, int64_t i1) {
        for (int64_t r = rows + i0; r < rows + i1; r++)
            bfp16_a_row(r, p.K, tile_, [&](int64_t, size_t off) { memset(A + off, 0, BFP16_BLOCK_BYTES); });
    });
    P.st->prep_ms += ms_since(h0);
    return true;
}

void npu_prefill::encode_row(const stream & s, int64_t K, int64_t t, const float * row) const {
    uint8_t * A = s.proj->a((int) ((t - s.t0) / BIG));
    bfp16_a_row((t - s.t0) % BIG, K, tile_, [&](int64_t cb, size_t off) { bfp16_encode_block(row + cb * 8, A + off); });
}

bool npu_prefill::attn_scores(pass & P, stream & s, int64_t b, std::string & err) {
    const int64_t Hkv = hp_.n_head_kv, D = hp_.head_dim, Ek = Hkv * D, Eq = hp_.n_head * D, T = P.T;
    const int64_t QB = BLOCK, D3 = 3 * D, M = 2 * QB, DB = D / 8, L = (b + 1) * QB;
    const clk::time_point h0 = clk::now();
    if (!s.sc->prepare(std::vector<int>((size_t) Hkv, scores_.at(L)), err, kv_once_ ? keys_ : std::vector<int>())) return false;
    npu_bfp16::batch & sc = *s.sc;
    const float * q = P.q.data(), * k = P.k.data();
    const int64_t nb_rows = kv_once_ ? M : M + L;  // the keys already sit in the shared buffers
    // Per kv head g. A: its two heads' 512 queries, rows [q | q | q - bfp16(q)].
    // B: the L keys, rows [k | k - bfp16(k) | k]. Rows past the prompt are zeros.
    pool_->parallel_for(Hkv * nb_rows, [&](int64_t i0, int64_t i1) {
        float lo[8], hi[8];
        size_t off[3 * 32];  // a row's block offsets, head dim <= 256
        for (int64_t i = i0; i < i1; i++) {
            const int64_t g = i / nb_rows, r = i % nb_rows;
            const float * src;
            uint8_t * dst;
            if (r < M) {
                const int64_t t = b * QB + r % QB;
                src = t < T ? q + t * Eq + (2 * g + r / QB) * D : nullptr;
                dst = sc.a((int) g);
                bfp16_a_row(r, D3, tile_, [&](int64_t cb, size_t o) { off[cb] = o; });
            } else {
                const int64_t kt = r - M;
                src = kt < T ? k + kt * Ek + g * D : nullptr;
                dst = sc.b((int) g);
                bfp16_b_row(kt, L, D3, tile_, [&](int64_t cb, size_t o) { off[cb] = o; });
            }
            // an A row is [x | x | lo]; a B row [x | lo | x]
            const int64_t at_x2 = r < M ? DB : 2 * DB, at_lo = r < M ? 2 * DB : DB;
            for (int64_t cb = 0; cb < DB; cb++) {
                uint8_t * o0 = dst + off[cb];
                if (!src) {
                    for (int64_t c : { cb, cb + DB, cb + 2 * DB }) memcpy(dst + off[c], zero_block, sizeof(zero_block));
                    continue;
                }
                bfp16_encode_block(src + cb * 8, o0);
                memcpy(dst + off[at_x2 + cb], o0, BFP16_BLOCK_BYTES);
                bfp16_decode_block(o0, hi);
                for (int j = 0; j < 8; j++) lo[j] = src[cb * 8 + j] - hi[j];
                bfp16_encode_block(lo, dst + off[at_lo + cb]);
            }
        }
    });
    P.st->attn_in_ms += ms_since(h0);
    P.st->calls += (int) Hkv;
    return true;
}

bool npu_prefill::attn_softmax(pass & P, stream & s, int64_t b, std::string & err) {
    const int64_t Hkv = hp_.n_head_kv, D = hp_.head_dim, Ek = Hkv * D, T = P.T;
    const int64_t QB = BLOCK, M = 2 * QB, PVN = 512, L = (b + 1) * QB;
    const float scale = 1.0f / std::sqrt((float) D);
    const clk::time_point h0 = clk::now();
    if (!s.pv->prepare(std::vector<int>((size_t) Hkv, pv_.at(L)), err, kv_once_ ? vt_ : std::vector<int>())) return false;
    // Only the first D of the 512 columns are real. In the --c-tiled layout
    // they are the first D / n array columns, one run at the front.
    s.pv->sync_c_prefix(c_tiled_ ? (size_t) (M * PVN / tile_.cols) * (D / tile_.n) * sizeof(float) : 0);
    npu_bfp16::batch & sc = *s.sc, & pv = *s.pv;
    const bfp16_tiling * tiled = c_tiled_ ? &tile_ : nullptr;
    const float * v = P.v.data();
    // The causal softmax, into probabilities times values' A (one row per
    // score row); and its B, v transposed: D rows of positions, the rest of
    // the 512 zeros (one run at the end, when each array column holds a
    // single n-row group).
    const int64_t tkb = tile_.k / 8, nsb = L / 8;
    std::vector<size_t> vrow((size_t) D);
    for (int64_t d = 0; d < D; d++) vrow[d] = bfp16_b_offset(d, 0, PVN, L, tile_);
    const bool zero_tail = PVN == (int64_t) tile_.n * tile_.cols && D % tile_.n == 0;
    const size_t pv_b_bytes = (size_t) (PVN * L / 8) * BFP16_BLOCK_BYTES, tail_at = bfp16_b_offset(D, 0, PVN, L, tile_);
    pool_->parallel_for(kv_once_ ? Hkv * M : Hkv * M + Hkv * nsb + Hkv, [&](int64_t i0, int64_t i1) {
        thread_local std::vector<float> srow, prow;
        srow.resize((size_t) L);
        prow.resize((size_t) L);
        float blk[8];
        for (int64_t i = i0; i < i1; i++) {
            if (i < Hkv * M) {
                const int64_t g = i / M, r = i % M, t = b * QB + r % QB;
                uint8_t * A = pv.a((int) g);
                const int64_t n = t < T ? t + 1 : 0;  // keys this query sees
                if (n > 0) {
                    // only the scores this query sees: the first n, in whole tiles
                    const float * S = sc.c((int) g) + r * L;
                    if (tiled) {
                        bfp16_c_row(sc.c((int) g), M, L, tile_, r, (n + tile_.n - 1) / tile_.n * tile_.n, srow.data());
                        S = srow.data();
                    }
                    softmax_row(S, n, scale, prow.data());
                    std::fill(prow.begin() + n, prow.begin() + ((n + 7) / 8) * 8, 0.0f);
                }
                bfp16_a_row(r, L, tile_, [&](int64_t cb, size_t o) {
                    if (cb * 8 < n) bfp16_encode_block(prow.data() + cb * 8, A + o);
                    else memcpy(A + o, zero_block, sizeof(zero_block));
                });
            } else if (i < Hkv * M + Hkv * nsb) {
                const int64_t j = i - Hkv * M, g = j / nsb, sb = j % nsb;
                uint8_t * B = pv.b((int) g) + ((sb / tkb) * tile_.n * tkb + (sb % tkb) * 16) * BFP16_BLOCK_BYTES;
                const int64_t valid = std::min<int64_t>(8, T - sb * 8);  // positions past the prompt are zeros
                const float * vs = valid > 0 ? v + sb * 8 * Ek + g * D : v;
                for (int64_t d = 0; d < D; d++) {
                    for (int64_t e = 0; e < 8; e++) blk[e] = e < valid ? vs[e * Ek + d] : 0.0f;
                    bfp16_encode_block(blk, B + vrow[d]);
                }
                if (!zero_tail)
                    for (int64_t d = D; d < PVN; d++)
                        memcpy(pv.b((int) g) + bfp16_b_offset(d, sb, PVN, L, tile_), zero_block, sizeof(zero_block));
            } else if (zero_tail) {
                const int64_t g = i - Hkv * M - Hkv * nsb;
                memset(pv.b((int) g) + tail_at, 0, pv_b_bytes - tail_at);
            }
        }
    });
    P.st->attn_soft_ms += ms_since(h0);
    P.st->calls += (int) Hkv;
    return true;
}

void npu_prefill::attn_output(pass & P, stream & s, int64_t b) {
    const int64_t Hq = hp_.n_head, D = hp_.head_dim, Eq = Hq * D, QB = BLOCK, M = 2 * QB, PVN = 512;
    const clk::time_point h0 = clk::now();
    const bfp16_tiling * tiled = c_tiled_ ? &tile_ : nullptr;
    npu_bfp16::batch & pv = *s.pv;
    // each of the block's tokens: its heads' results, gathered into a row and
    // encoded as the output projection's input
    pool_->parallel_for(std::min(P.T, (b + 1) * QB) - b * QB, [&](int64_t i0, int64_t i1) {
        thread_local std::vector<float> row;
        row.resize((size_t) Eq);
        for (int64_t i = i0; i < i1; i++) {
            for (int64_t h = 0; h < Hq; h++) {
                const float * O = pv.c((int) (h / 2));
                const int64_t r = (h % 2) * QB + i;  // kv head h/2's rows: its two heads' queries
                if (tiled) bfp16_c_row(O, M, PVN, tile_, r, D, row.data() + h * D);
                else std::copy(O + r * PVN, O + r * PVN + D, row.data() + h * D);
            }
            encode_row(s, Eq, b * QB + i, row.data());
        }
    });
    P.st->attn_out_ms += ms_since(h0);
}

void npu_prefill::attention_host(const float * q, const float * k, const float * v, int64_t t0, int64_t t1, float * a) {
    const int64_t Hq = hp_.n_head, Hkv = hp_.n_head_kv, D = hp_.head_dim, group = Hq / Hkv, Ek = Hkv * D;
    const float scale = 1.0f / std::sqrt((float) D);
    const int64_t QB = 16;                     // queries per task
    const int64_t nqb = (t1 - t0 + QB - 1) / QB;
    // larger query blocks do more work (causal), so hand them out first
    pool_->parallel_for(Hq * nqb, [&](int64_t b, int64_t e) {
        std::vector<float> p((size_t) t1);
        for (int64_t task = b; task < e; task++) {
            const int64_t h = task % Hq, qb = nqb - 1 - task / Hq, g = h / group;
            for (int64_t t = t0 + qb * QB; t < std::min(t1, t0 + (qb + 1) * QB); t++) {
                const float * qt = q + (t * Hq + h) * D;
                float mx = -INFINITY;
                for (int64_t s = 0; s <= t; s++) {
                    p[s] = dot(qt, k + s * Ek + g * D, D) * scale;
                    mx = std::max(mx, p[s]);
                }
                float sum = 0.0f;
                for (int64_t s = 0; s <= t; s++) { p[s] = std::exp(p[s] - mx); sum += p[s]; }
                const float inv = 1.0f / sum;
                float * ot = a + (t * Hq + h) * D;
                std::fill(ot, ot + D, 0.0f);
                for (int64_t s = 0; s <= t; s++) {
                    const float w = p[s] * inv;
                    const float * vs = v + s * Ek + g * D;
                    for (int64_t i = 0; i < D; i++) ot[i] += w * vs[i];
                }
            }
        }
    });
}

// A stream's steps through layer il. Each is host work followed by the NPU
// work it feeds, started and left running:
//   0              norm -> q/k/v's input                         [q/k/v]
//   1              read q/k/v; block b0's scores' operands        [scores b0]
//   2 + 2i         block i's softmax -> values' operands          [values]
//   3 + 2i         block i's output -> o's input; then the next
//                  block's scores' operands                       [scores]
//                  or, after the last block                       [o]
//   ka = 2 + 2n    read o (residual); norm -> gate/up's input     [gate/up]
//   ka + 1         read gate/up: SiLU(gate) * up -> down's input  [down]
//   ka + 2         read down (residual)                           done
// With attention on the host, step 1 reads q/k/v, runs attention, encodes
// o's input and starts o, and ka = 2. On the GPU, step 1 reads q/k/v and
// starts attention on the GPU; step 2 waits for it, encodes o's input and
// starts o; ka = 3.
bool npu_prefill::step(pass & P, stream & s, int il, int k, npu_bfp16::batch *& npu, bool & done, std::string & err) {
    const layer & L = L_[il];
    const int64_t E = hp_.n_embd, Hq = hp_.n_head, Hkv = hp_.n_head_kv, D = hp_.head_dim, F = hp_.n_ff;
    const int64_t Eq = Hq * D, Ek = Hkv * D;
    const float eps = hp_.rms_eps;
    const int64_t b0 = s.t0 / BLOCK, nblk = (s.t1 - s.t0 + BLOCK - 1) / BLOCK;
    const int ka = P.attn_npu ? (int) (2 + 2 * nblk) : P.attn_gpu ? 3 : 2;
    const int64_t rows = s.t1 - s.t0;
    const bfp16_tiling * tiled = c_tiled_ ? &tile_ : nullptr;
    npu = nullptr;
    done = false;

    // norm(x) * w for s's rows, encoded as projection p's input, then p started
    auto norm_into = [&](const std::vector<float> & w, const proj & p) {
        if (!prepare_proj(P, s, p, err)) return false;
        const clk::time_point h0 = clk::now();
        pool_->parallel_for(rows, [&](int64_t i0, int64_t i1) {
            thread_local std::vector<float> row;
            row.resize((size_t) E);
            for (int64_t t = s.t0 + i0; t < s.t0 + i1; t++) {
                rmsnorm_row(P.x.data() + t * E, w.data(), E, eps, row.data());
                encode_row(s, E, t, row.data());
            }
        });
        P.st->norm_ms += ms_since(h0);
        P.st->calls += s.calls();
        npu = s.proj.get();
        return true;
    };
    // fn(t, output row) over s's rows of the projection that just ran; the
    // time goes to read_ms and to that projection's own column
    auto read = [&](const proj & p, double & ms, const auto & fn) {
        const clk::time_point h0 = clk::now();
        pool_->parallel_for(rows, [&](int64_t i0, int64_t i1) {
            thread_local std::vector<float> scratch;
            scratch.resize((size_t) p.N);
            for (int64_t i = i0; i < i1; i++) {
                const int j = (int) (i / BIG);
                fn(s.t0 + i, out_view{ s.proj->c(j), s.call_m(j), p.N, tiled, p.mode }.row(i % BIG, scratch.data()));
            }
        });
        const double took = ms_since(h0);
        P.st->read_ms += took;
        ms += took;
    };
    auto add_residual = [&](const proj & p, double & ms) {
        read(p, ms, [&](int64_t t, const float * y) {
            float * xt = P.x.data() + t * E;
            for (int64_t i = 0; i < E; i++) xt[i] += y[i];
        });
    };
    auto start_o = [&]() {
        if (!prepare_proj(P, s, L.o, err)) return false;
        P.st->calls += s.calls();
        npu = s.proj.get();
        return true;
    };

    if (k == 0) return norm_into(L.attn_norm, L.qkv);
    if (k == 1) {
        // q/k/v: split, per-head norm and rotary on q and k, and the cache rows
        read(L.qkv, P.st->read_qkv_ms, [&](int64_t t, const float * y) {
            float * qt = P.q.data() + t * Eq, * kt = P.k.data() + t * Ek, * vt = P.v.data() + t * Ek;
            std::copy(y, y + Eq, qt);
            std::copy(y + Eq, y + Eq + Ek, kt);
            std::copy(y + Eq + Ek, y + Eq + 2 * Ek, vt);
            for (int64_t hh = 0; hh < Hq; hh++)  norm_rope_head(qt + hh * D, L.q_norm.data(), D, eps, P.rope.data() + t * D);
            for (int64_t hh = 0; hh < Hkv; hh++) norm_rope_head(kt + hh * D, L.k_norm.data(), D, eps, P.rope.data() + t * D);
            ggml_fp32_to_fp16_row(kt, (ggml_fp16_t *) P.out->k[il].data() + t * Ek, Ek);
            ggml_fp32_to_fp16_row(vt, (ggml_fp16_t *) P.out->v[il].data() + t * Ek, Ek);
        });
        if (P.attn_npu) {
            if (kv_once_ && !encode_kv(P, s, err)) return false;
            if (!attn_scores(P, s, b0, err)) return false;
            npu = s.sc.get();
            return true;
        }
        if (P.attn_gpu) {
            // keys and values as the cache holds them (float16), rows 0 to t1:
            // the earlier streams' rows were read a step ago
            s.gpu_ticket = gpu_->start(P.q.data(), P.out->k[il].data(), P.out->v[il].data(), s.t0, s.t1, P.a.data());
            return true;
        }
        const clk::time_point h0 = clk::now();
        attention_host(P.q.data(), P.k.data(), P.v.data(), s.t0, s.t1, P.a.data());
        pool_->parallel_for(rows, [&](int64_t i0, int64_t i1) {
            for (int64_t t = s.t0 + i0; t < s.t0 + i1; t++) encode_row(s, Eq, t, P.a.data() + t * Eq);
        });
        P.st->attn_soft_ms += ms_since(h0);
        return start_o();
    }
    if (P.attn_gpu && k == 2) {
        clk::time_point h0 = clk::now();
        if (!gpu_->wait(s.gpu_ticket, err)) return false;
        P.st->gpu_wait_ms += ms_since(h0);
        h0 = clk::now();
        pool_->parallel_for(rows, [&](int64_t i0, int64_t i1) {
            for (int64_t t = s.t0 + i0; t < s.t0 + i1; t++) encode_row(s, Eq, t, P.a.data() + t * Eq);
        });
        P.st->attn_out_ms += ms_since(h0);
        return start_o();
    }
    if (k < ka) {
        const int64_t i = (k - 2) / 2, b = b0 + i;
        if (k % 2 == 0) {
            if (!attn_softmax(P, s, b, err)) return false;
            npu = s.pv.get();
            return true;
        }
        attn_output(P, s, b);
        if (i + 1 < nblk) {
            if (!attn_scores(P, s, b + 1, err)) return false;
            npu = s.sc.get();
            return true;
        }
        return start_o();
    }
    if (k == ka) {
        add_residual(L.o, P.st->read_o_ms);
        return norm_into(L.ffn_norm, L.gate_up);
    }
    if (k == ka + 1) {
        // SiLU(gate) * up, encoded as down's input: computed here, or already
        // on the NPU in mode 2. Down's buffers are gate/up's, whose input is
        // spent; its output is read here, before down runs.
        if (!prepare_proj(P, s, L.down, err)) return false;
        if (L.gate_up.mode == 2)
            read(L.gate_up, P.st->read_gu_ms, [&](int64_t t, const float * g) { encode_row(s, F, t, g); });
        else
            read(L.gate_up, P.st->read_gu_ms, [&](int64_t t, const float * y) {
                thread_local std::vector<float> g;
                g.resize((size_t) F);
                swiglu_row(y, y + F, F, g.data());
                encode_row(s, F, t, g.data());
            });
        P.st->calls += s.calls();
        npu = s.proj.get();
        return true;
    }
    add_residual(L.down, P.st->read_down_ms);
    done = true;
    return true;
}

bool npu_prefill::prefill(const std::vector<int32_t> & toks, qwen3_prefill_out & out, npu_prefill_stats & st,
                          std::string & err) {
    const clk::time_point t_start = clk::now();
    st = {};
    npu_.sync_ms = npu_.sync_in_ms = npu_.sync_out_ms = 0;
    const int64_t T = (int64_t) toks.size();
    const int64_t E = hp_.n_embd, Hq = hp_.n_head, Hkv = hp_.n_head_kv, D = hp_.head_dim;
    const int64_t Eq = Hq * D, Ek = Hkv * D, half = D / 2;

    pass P;
    P.T = T;
    P.out = &out;
    P.st = &st;
    P.attn_npu = attention_on_npu(T);
    P.attn_gpu = attn == attn_where::gpu;
    if (P.attn_gpu && !gpu_) {
        gpu_ = std::make_unique<gpu_attention>();
        if (!gpu_->init(gpu_device, Hq, Hkv, D, err)) { gpu_.reset(); return false; }
    }
    if (gpu_) gpu_->take_times();
    P.x.resize((size_t) (T * E));
    P.q.resize((size_t) (T * Eq));
    P.k.resize((size_t) (T * Ek));
    P.v.resize((size_t) (T * Ek));
    if (!P.attn_npu) P.a.resize((size_t) (T * Eq));
    out.k.assign(hp_.n_layer, std::vector<uint16_t>((size_t) (T * Ek)));
    out.v.assign(hp_.n_layer, std::vector<uint16_t>((size_t) (T * Ek)));
    out.logits_last.clear();

    clk::time_point h0 = clk::now();
    pool_->parallel_for(T, [&](int64_t b, int64_t e) {
        for (int64_t t = b; t < e; t++) model_->embedding(toks[t], P.x.data() + t * E);
    });
    P.rope.resize((size_t) (T * D));
    {
        std::vector<float> freq((size_t) half);
        for (int64_t j = 0; j < half; j++) freq[j] = std::pow(hp_.rope_base, -2.0f * (float) j / (float) D);
        pool_->parallel_for(T, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++)
                for (int64_t j = 0; j < half; j++) {
                    P.rope[t * D + j]        = std::cos((float) t * freq[j]);
                    P.rope[t * D + half + j] = std::sin((float) t * freq[j]);
                }
        });
    }
    st.other_ms += ms_since(h0);

    // the streams: whole 512-query blocks, split as evenly as they go
    const int64_t nb = (T + BLOCK - 1) / BLOCK;
    const int S = (int) std::max<int64_t>(1, std::min<int64_t>(max_streams, nb));
    while ((int) streams_.size() < S) {
        streams_.emplace_back();
        stream & s = streams_.back();
        s.proj = std::make_unique<npu_bfp16::batch>(npu_);
        s.sc   = std::make_unique<npu_bfp16::batch>(npu_);
        s.pv   = std::make_unique<npu_bfp16::batch>(npu_);
    }
    for (int i = 0; i < S; i++) {
        stream & s = streams_[i];
        s.t0 = std::min(T, nb * i / S * BLOCK);
        s.t1 = std::min(T, nb * (i + 1) / S * BLOCK);
        if (!s.proj->reserve(s.calls(), a_bytes_, 0, c_bytes_, err)) return false;
    }
    st.streams = S;

    // Layer by layer, the streams take turns: each waits for its own NPU work,
    // does its next host step, and starts its next NPU work, which queues
    // behind the other stream's. So the host works on one stream while the
    // NPU runs the other's.
    for (int il = 0; il < hp_.n_layer; il++) {
        std::vector<int> k((size_t) S, 0);
        std::vector<npu_bfp16::batch *> pending((size_t) S, nullptr);
        std::vector<char> finished((size_t) S, 0);
        for (int left = S; left > 0;) {
            for (int i = 0; i < S; i++) {
                if (finished[i]) continue;
                if (pending[i]) {
                    h0 = clk::now();
                    if (!pending[i]->wait(err)) return false;
                    st.wait_ms += ms_since(h0);
                }
                bool done = false;
                if (!step(P, streams_[i], il, k[i]++, pending[i], done, err)) return false;
                if (pending[i]) {
                    h0 = clk::now();
                    if (!pending[i]->start(err)) return false;
                    st.start_ms += ms_since(h0);
                }
                if (done) { finished[i] = 1; left--; }
            }
        }
    }
    st.sync_ms = npu_.sync_ms;
    st.sync_in_ms = npu_.sync_in_ms;
    st.sync_out_ms = npu_.sync_out_ms;
    if (P.attn_gpu) {
        const gpu_attention::times g = gpu_->take_times();
        st.gpu_in_ms = g.in_ms;
        st.gpu_run_ms = g.run_ms;
        st.gpu_out_ms = g.out_ms;
        st.gpu_build_ms = g.build_ms;
        st.gpu_out_imported = gpu_->out_imported();
    }
    st.total_ms = ms_since(t_start);
    out.ms = st.total_ms;
    return true;
}
