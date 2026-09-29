#include "qwen3_ref.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

#ifdef _MSC_VER
#define FSEEK64 _fseeki64
#else
#define FSEEK64 fseeko
#endif

namespace {

void parallel_for(int64_t n, int nth, const std::function<void(int64_t, int64_t)> & fn) {
    nth = (int) std::max<int64_t>(1, std::min<int64_t>(nth, n));
    if (nth == 1) { fn(0, n); return; }
    const int64_t chunk = (n + nth - 1) / nth;
    std::vector<std::thread> pool;
    for (int t = 1; t < nth; t++) {
        const int64_t b = t * chunk, e = std::min(n, b + chunk);
        if (b < e) pool.emplace_back(fn, b, e);
    }
    fn(0, std::min(n, chunk));
    for (auto & t : pool) t.join();
}

inline float dot(const float * a, const float * b, int64_t n) {
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// y[t][o] = sum_k W[o][k] * x[t][k]
void matmul(const float * W, int64_t O, int64_t K, const float * x, int64_t T, float * y, int nth) {
    parallel_for(O, nth, [&](int64_t b, int64_t e) {
        for (int64_t t = 0; t < T; t++) {
            const float * xt = x + t * K;
            for (int64_t o = b; o < e; o++) {
                y[t * O + o] = dot(W + o * K, xt, K);
            }
        }
    });
}

// x[t][n] scaled by 1/sqrt(mean(x^2) + eps), times w[n]
void rmsnorm(const float * x, const float * w, int64_t T, int64_t n, float eps, float * y) {
    for (int64_t t = 0; t < T; t++) {
        const float * xt = x + t * n;
        double ss = 0.0;
        for (int64_t i = 0; i < n; i++) ss += (double) xt[i] * xt[i];
        const float scale = 1.0f / std::sqrt((float) (ss / n) + eps);
        for (int64_t i = 0; i < n; i++) y[t * n + i] = xt[i] * scale * w[i];
    }
}

// Rotary encoding, half-split pairs (j, j + d/2), angle pos * base^(-2j/d),
// applied to every head of x[t][h][d] with position = t.
void rope_neox(float * x, int64_t T, int64_t H, int64_t d, float base) {
    const int64_t half = d / 2;
    std::vector<float> freq(half);
    for (int64_t j = 0; j < half; j++) freq[j] = std::pow(base, -2.0f * (float) j / (float) d);
    for (int64_t t = 0; t < T; t++) {
        for (int64_t h = 0; h < H; h++) {
            float * v = x + (t * H + h) * d;
            for (int64_t j = 0; j < half; j++) {
                const float th = (float) t * freq[j];
                const float c = std::cos(th), s = std::sin(th);
                const float x0 = v[j], x1 = v[j + half];
                v[j]        = x0 * c - x1 * s;
                v[j + half] = x0 * s + x1 * c;
            }
        }
    }
}

float round_bf16(float x);

// Causal grouped-query attention. q[t][Hq][d], k/v[t][Hkv][d] -> o[t][Hq][d].
// scores_bf16 rounds each score to bf16 before it is scaled.
void attention(const float * q, const float * k, const float * v, int64_t T, int64_t Hq, int64_t Hkv, int64_t d,
               float * o, int nth, bool scores_bf16 = false) {
    const float scale = 1.0f / std::sqrt((float) d);
    const int64_t group = Hq / Hkv;
    parallel_for(T, nth, [&](int64_t b, int64_t e) {
        std::vector<float> p(T);
        for (int64_t t = b; t < e; t++) {
            for (int64_t h = 0; h < Hq; h++) {
                const int64_t g = h / group;
                const float * qt = q + (t * Hq + h) * d;
                float mx = -INFINITY;
                for (int64_t s = 0; s <= t; s++) {
                    const float sc = dot(qt, k + (s * Hkv + g) * d, d);
                    p[s] = (scores_bf16 ? round_bf16(sc) : sc) * scale;
                    mx = std::max(mx, p[s]);
                }
                double sum = 0.0;
                for (int64_t s = 0; s <= t; s++) { p[s] = std::exp(p[s] - mx); sum += p[s]; }
                float * ot = o + (t * Hq + h) * d;
                std::fill(ot, ot + d, 0.0f);
                for (int64_t s = 0; s <= t; s++) {
                    const float w = (float) (p[s] / sum);
                    const float * vs = v + (s * Hkv + g) * d;
                    for (int64_t i = 0; i < d; i++) ot[i] += w * vs[i];
                }
            }
        }
    });
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// ---- rounding that stands in for an NPU datapath ----------------------------

float round_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest, ties to even
    u &= 0xffff0000u;
    memcpy(&x, &u, 4);
    return x;
}

// bfp16ebs8 as AIE2P stores it (mlir-aie programming_examples/ml/block_datatypes/
// helper.h): eight consecutive values share the largest of their eight
// exponents, and each keeps a signed 8-bit mantissa, so every value in the
// block is a multiple of 2^(shared exponent - 6). The helper's comment and its
// two's-complement shifts say an AIE2P core converts by truncation, which on
// two's complement is rounding down; `floor_mode` reproduces that.
//
// Rounding to nearest has two forms, for the one case where a block's largest
// value rounds up to +128. The host converter (kernels/bfp16_gemm's
// floatToBfp16Rne, and the packer) saturates it at 127, which costs only that
// value. The core under aie::rounding_mode::conv_even raises the shared
// exponent and rounds the whole block again, measured on the NPU 2026-09-26;
// `core` selects that form.
void round_bfp16(float * x, int64_t n, bool floor_mode, bool core = false) {
    for (int64_t b = 0; b < n; b += 8) {
        const int64_t e = std::min<int64_t>(n, b + 8);
        int maxexp = 0;
        for (int64_t i = b; i < e; i++) {
            uint32_t u;
            memcpy(&u, &x[i], 4);
            maxexp = std::max(maxexp, (int) ((u >> 23) & 0xffu));
        }
        if (maxexp == 0xff) continue;
        for (int bump = 0; bump < 2; bump++) {
            const float step = std::ldexp(1.0f, maxexp + bump - 127 - 6);
            bool over = false;
            float q[8];
            for (int64_t i = b; i < e; i++) {
                const float v = x[i] / step;
                q[i - b] = floor_mode ? std::floor(v) : std::nearbyint(v);
                over |= q[i - b] > 127.0f;
            }
            if (over && core && bump == 0) continue;
            for (int64_t i = b; i < e; i++) x[i] = std::min(127.0f, std::max(-128.0f, q[i - b])) * step;
            break;
        }
    }
}

// Q8_0 as ggml's quantize_row_q8_0_ref writes it and its dequant reads it back.
void round_q8_0(float * x, int64_t n) {
    for (int64_t b = 0; b < n; b += 32) {
        const int64_t e = std::min<int64_t>(n, b + 32);
        float amax = 0.0f;
        for (int64_t i = b; i < e; i++) amax = std::max(amax, std::fabs(x[i]));
        const float d  = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const float dh = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
        for (int64_t i = b; i < e; i++) x[i] = std::roundf(x[i] * id) * dh;
    }
}

// Weights, one row of K at a time, as they would be converted when the model
// loads. A host-side conversion can always round to nearest.
void round_weights(std::vector<float> & w, int64_t K, qwen3_rounding r) {
    const int64_t rows = (int64_t) w.size() / K;
    switch (r) {
        case qwen3_rounding::bf16:
            for (float & v : w) v = round_bf16(v);
            break;
        case qwen3_rounding::bfp16:
        case qwen3_rounding::bfp16_floor:
            for (int64_t i = 0; i < rows; i++) round_bfp16(w.data() + i * K, K, false);
            break;
        case qwen3_rounding::none:
        case qwen3_rounding::q8_0:
            break;
    }
}

// Activations, one token's K values at a time, as the datapath sees them.
void round_acts(float * x, int64_t T, int64_t K, qwen3_rounding r) {
    for (int64_t t = 0; t < T; t++) {
        float * xt = x + t * K;
        switch (r) {
            case qwen3_rounding::none:        break;
            case qwen3_rounding::bf16:        for (int64_t i = 0; i < K; i++) xt[i] = round_bf16(xt[i]); break;
            case qwen3_rounding::q8_0:        round_q8_0(xt, K); break;
            case qwen3_rounding::bfp16:       round_bfp16(xt, K, false); break;
            case qwen3_rounding::bfp16_floor: round_bfp16(xt, K, true); break;
        }
    }
}

// y = W x with the running sum rounded after every `kc` along K, the way a
// kernel that keeps its output tile narrower than float32 does
void matmul_acc(const float * W, int64_t O, int64_t K, const float * x, int64_t T, float * y, int nth,
                qwen3_prefill_opts::acc_type acc, int64_t kc) {
    parallel_for(T, nth, [&](int64_t b, int64_t e) {
        for (int64_t t = b; t < e; t++) {
            const float * xt = x + t * K;
            float * yt = y + t * O;
            std::fill(yt, yt + O, 0.0f);
            for (int64_t c = 0; c < K; c += kc) {
                const int64_t n = std::min(kc, K - c);
                for (int64_t o = 0; o < O; o++) yt[o] += dot(W + o * K + c, xt + c, n);
                if (acc == qwen3_prefill_opts::acc_type::bf16) {
                    for (int64_t o = 0; o < O; o++) yt[o] = round_bf16(yt[o]);
                } else {
                    round_bfp16(yt, O, false, true);  // the core converts its own running sums
                }
            }
        }
    });
}

// matmul with the datapath's rounding applied to copies of both inputs;
// `which` is this projection's qwen3_prefill_opts bit
void matmul_r(std::vector<float> & W, int64_t O, int64_t K, const float * x, int64_t T, float * y,
              int nth, const qwen3_prefill_opts & opts, unsigned which) {
    const qwen3_rounding r = (opts.bf16_projections & which) ? qwen3_rounding::bf16 : opts.rounding;
    if (r == qwen3_rounding::none) { matmul(W.data(), O, K, x, T, y, nth); return; }
    if (!opts.keep_weights) round_weights(W, K, r);
    std::vector<float> xr(x, x + T * K);
    if (!opts.keep_acts) round_acts(xr.data(), T, K, r);
    if (opts.acc == qwen3_prefill_opts::acc_type::f32) {
        matmul(W.data(), O, K, xr.data(), T, y, nth);
    } else {
        matmul_acc(W.data(), O, K, xr.data(), T, y, nth, opts.acc, opts.acc_chunk);
    }
}

// Attention with both products' inputs rounded: q and k along the head
// dimension for the scores, the probabilities and v along positions for the
// output.
void attention_r(const float * q, const float * k, const float * v, int64_t T, int64_t Hq, int64_t Hkv, int64_t d,
                 float * o, int nth, qwen3_rounding r, bool scores, bool values, bool split, bool scores_bf16,
                 const qwen3_prefill_opts * sm = nullptr) {
    const qwen3_rounding rs = scores ? r : qwen3_rounding::none, rv = values ? r : qwen3_rounding::none;
    const bool npu_sm = sm && sm->softmax_npu;
    // softmax_npu: q carries the scale, so the scores arrive in log2 units
    std::vector<float> qsc;
    if (npu_sm) {
        const float c2 = (float) (1.4426950408889634 / std::sqrt((double) d));
        qsc.assign(q, q + T * Hq * d);
        for (float & e : qsc) e *= c2;
        q = qsc.data();
    }
    std::vector<float> qr(q, q + T * Hq * d), kr(k, k + T * Hkv * d);
    round_acts(qr.data(), T * Hq, d, rs);
    round_acts(kr.data(), T * Hkv, d, rs);
    // the low parts: what rounding left behind, rounded in turn
    std::vector<float> ql, kl;
    if (split) {
        ql.resize(qr.size());
        kl.resize(kr.size());
        for (size_t i = 0; i < qr.size(); i++) ql[i] = q[i] - qr[i];
        for (size_t i = 0; i < kr.size(); i++) kl[i] = k[i] - kr[i];
        round_acts(ql.data(), T * Hq, d, r);
        round_acts(kl.data(), T * Hkv, d, r);
    }
    // v transposed so that each (head, dim) column's positions are contiguous
    std::vector<float> vt((size_t) Hkv * d * T);
    for (int64_t s = 0; s < T; s++)
        for (int64_t j = 0; j < Hkv * d; j++) vt[j * T + s] = v[s * Hkv * d + j];
    round_acts(vt.data(), Hkv * d, T, rv);

    const float scale = npu_sm ? 1.0f : 1.0f / std::sqrt((float) d);
    const int64_t group = Hq / Hkv;

    // softmax_npu's upper bounds, over the keys each query sees (0..t), from
    // the unrounded keys: the largest |k|, and each dimension's range
    std::vector<float> knmax, kmax, kmin;
    std::vector<uint8_t> fb;  // rows that fall back
    if (npu_sm) {
        knmax.resize((size_t) (T * Hkv));
        kmax.resize((size_t) (T * Hkv * d));
        kmin.resize((size_t) (T * Hkv * d));
        fb.assign((size_t) (T * Hq), 0);
        for (int64_t s = 0; s < T; s++) {
            for (int64_t g = 0; g < Hkv; g++) {
                const float * ks = k + (s * Hkv + g) * d;
                const size_t j = (size_t) (s * Hkv + g);
                const float n = std::sqrt(dot(ks, ks, d));
                knmax[j] = s ? std::max(knmax[j - Hkv], n) : n;
                for (int64_t i = 0; i < d; i++) {
                    kmax[j * d + i] = s ? std::max(kmax[(j - Hkv) * d + i], ks[i]) : ks[i];
                    kmin[j * d + i] = s ? std::min(kmin[(j - Hkv) * d + i], ks[i]) : ks[i];
                }
            }
        }
    }
    std::mutex stats_mu;
    softmax_bias_stats stats;

    parallel_for(T, nth, [&](int64_t b, int64_t e) {
        std::vector<float> p(T);
        softmax_bias_stats st;
        for (int64_t t = b; t < e; t++) {
            for (int64_t h = 0; h < Hq; h++) {
                const int64_t g = h / group;
                const float * qt = qr.data() + (t * Hq + h) * d;
                float mx = -INFINITY;
                for (int64_t s = 0; s <= t; s++) {
                    const float * ks = kr.data() + (s * Hkv + g) * d;
                    float sc = dot(qt, ks, d);
                    if (split) {
                        sc += dot(qt, kl.data() + (s * Hkv + g) * d, d) + dot(ql.data() + (t * Hq + h) * d, ks, d);
                    }
                    p[s] = (scores_bf16 ? round_bf16(sc) : sc) * scale;
                    mx = std::max(mx, p[s]);
                }
                float * ot = o + (t * Hq + h) * d;
                if (npu_sm) {
                    // the bounds, from the unrounded (scaled) query
                    const float * qf = q + (t * Hq + h) * d;
                    const size_t j = (size_t) (t * Hkv + g);
                    double box = 0.0;
                    for (int64_t i = 0; i < d; i++)
                        box += std::max(qf[i] * kmax[j * d + i], qf[i] * kmin[j * d + i]);
                    const double cs = std::sqrt((double) dot(qf, qf, d)) * knmax[j];
                    const double up = std::min(cs, box);
                    double lo = dot(qf, k + g * d, d);
                    lo = std::max(lo, (double) dot(qf, k + (t * Hkv + g) * d, d));
                    if (t > 0) lo = std::max(lo, (double) dot(qf, k + ((t - 1) * Hkv + g) * d, d));
                    const double w = up - lo;
                    auto bucket = [](std::vector<int64_t> & hist, double x) {
                        hist[(size_t) std::min<double>(softmax_bias_stats::NB - 1, std::max(0.0, x))]++;
                    };
                    bucket(st.up_gap, up - mx);
                    bucket(st.lo_gap, mx - lo);
                    bucket(st.window, w);
                    st.worst_up = std::max(st.worst_up, up - mx);
                    st.worst_lo = std::max(st.worst_lo, mx - lo);
                    st.worst_window = std::max(st.worst_window, w);
                    st.cs_tighter += cs < box;
                    st.rows++;
                    if (w <= sm->bias_window) {
                        // -b travels as one bfp16 block [-b, 0 ...] in the scores' K
                        float nb[8] = { (float) (-(up + lo) / 2) };
                        round_bfp16(nb, 8, false);
                        for (int64_t s = 0; s <= t; s++) {
                            float x = std::exp2(p[s] + nb[0]);
                            if (x < FLT_MIN) x = 0.0f;  // the core flushes denormals
                            p[s] = sm->exp_bf16 ? round_bf16(x) : x;
                        }
                        round_acts(p.data(), 1, t + 1, rv);
                        float sum = 0.0f;  // what P.V's column of ones adds up, in float32
                        for (int64_t s = 0; s <= t; s++) sum += p[s];
                        for (int64_t i = 0; i < d; i++) ot[i] = dot(p.data(), vt.data() + (g * d + i) * T, t + 1) / sum;
                        continue;
                    }
                    st.rows_fallback++;
                    fb[(size_t) (t * Hq + h)] = 1;
                }
                // the host softmax as today (with softmax_npu, from scores in log2 units)
                double sum = 0.0;
                if (npu_sm) {
                    for (int64_t s = 0; s <= t; s++) { p[s] = std::exp2(p[s] - mx); sum += p[s]; }
                } else {
                    for (int64_t s = 0; s <= t; s++) { p[s] = std::exp(p[s] - mx); sum += p[s]; }
                }
                for (int64_t s = 0; s <= t; s++) p[s] = (float) (p[s] / sum);
                round_acts(p.data(), 1, t + 1, rv);
                for (int64_t i = 0; i < d; i++) {
                    ot[i] = dot(p.data(), vt.data() + (g * d + i) * T, t + 1);
                }
            }
        }
        if (npu_sm) {
            std::lock_guard<std::mutex> lk(stats_mu);
            stats.merge(st);
        }
    });

    if (npu_sm && sm->bias_stats) {
        // a core's tile is 128 queries of one head; a scores call is 512
        // queries of both heads that share a kv head
        for (int64_t h = 0; h < Hq; h++)
            for (int64_t t0 = 0; t0 < T; t0 += 128) {
                bool any = false;
                for (int64_t t = t0; t < std::min(T, t0 + 128); t++) any |= fb[(size_t) (t * Hq + h)] != 0;
                stats.tiles++;
                stats.tiles_fallback += any;
            }
        for (int64_t g = 0; g < Hkv; g++)
            for (int64_t t0 = 0; t0 < T; t0 += 512) {
                bool any = false;
                for (int64_t h = g * group; h < (g + 1) * group; h++)
                    for (int64_t t = t0; t < std::min(T, t0 + 512); t++) any |= fb[(size_t) (t * Hq + h)] != 0;
                stats.calls++;
                stats.calls_fallback += any;
            }
        sm->bias_stats->merge(stats);
    }
}

} // namespace

void softmax_bias_stats::merge(const softmax_bias_stats & o) {
    rows += o.rows; rows_fallback += o.rows_fallback;
    tiles += o.tiles; tiles_fallback += o.tiles_fallback;
    calls += o.calls; calls_fallback += o.calls_fallback;
    cs_tighter += o.cs_tighter;
    for (int i = 0; i < NB; i++) { up_gap[i] += o.up_gap[i]; lo_gap[i] += o.lo_gap[i]; window[i] += o.window[i]; }
    worst_up = std::max(worst_up, o.worst_up);
    worst_lo = std::max(worst_lo, o.worst_lo);
    worst_window = std::max(worst_window, o.worst_window);
}

const char * qwen3_rounding_name(qwen3_rounding r) {
    switch (r) {
        case qwen3_rounding::none:        return "float32";
        case qwen3_rounding::bf16:        return "bf16";
        case qwen3_rounding::q8_0:        return "q8_0 acts";
        case qwen3_rounding::bfp16:       return "bfp16 nearest";
        case qwen3_rounding::bfp16_floor: return "bfp16 floor";
    }
    return "?";
}

qwen3_ref::~qwen3_ref() {
    if (f_)    fclose(f_);
    if (gctx_) ggml_free(gctx_);
    if (gguf_) gguf_free(gguf_);
}

bool qwen3_ref::find(const char * name, tref & out, std::string & err) const {
    const int64_t id = gguf_find_tensor(gguf_, name);
    if (id < 0) { err = std::string("tensor not found: ") + name; return false; }
    const ggml_tensor * t = ggml_get_tensor(gctx_, name);
    out.id     = id;
    out.type   = (int) gguf_get_tensor_type(gguf_, id);
    out.offset = gguf_get_tensor_offset(gguf_, id);
    out.size   = gguf_get_tensor_size(gguf_, id);
    out.ne0    = t->ne[0];
    out.ne1    = t->ne[1];
    return true;
}

bool qwen3_ref::load(const std::string & path, std::string & err) {
    gguf_init_params ip = { /*no_alloc*/ true, &gctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), ip);
    if (!gguf_) { err = "gguf_init_from_file failed"; return false; }
    data_off_ = gguf_get_data_offset(gguf_);

    f_ = fopen(path.c_str(), "rb");
    if (!f_) { err = "cannot open " + path; return false; }

    auto u32 = [&](const char * key, int32_t & v) {
        const int64_t k = gguf_find_key(gguf_, key);
        if (k < 0) { err = std::string("key not found: ") + key; return false; }
        v = (int32_t) gguf_get_val_u32(gguf_, k);
        return true;
    };
    auto f32 = [&](const char * key, float & v) {
        const int64_t k = gguf_find_key(gguf_, key);
        if (k < 0) { err = std::string("key not found: ") + key; return false; }
        v = gguf_get_val_f32(gguf_, k);
        return true;
    };
    {
        const int64_t k = gguf_find_key(gguf_, "general.architecture");
        if (k < 0 || std::string(gguf_get_val_str(gguf_, k)) != "qwen3") {
            err = "not a qwen3 model";
            return false;
        }
    }
    if (!u32("qwen3.embedding_length",      hp_.n_embd))    return false;
    if (!u32("qwen3.block_count",           hp_.n_layer))   return false;
    if (!u32("qwen3.attention.head_count",  hp_.n_head))    return false;
    if (!u32("qwen3.attention.head_count_kv", hp_.n_head_kv)) return false;
    if (!u32("qwen3.attention.key_length",  hp_.head_dim))  return false;
    if (!u32("qwen3.feed_forward_length",   hp_.n_ff))      return false;
    if (!f32("qwen3.attention.layer_norm_rms_epsilon", hp_.rms_eps)) return false;
    if (!f32("qwen3.rope.freq_base",        hp_.rope_base)) return false;

    if (!find("token_embd.weight", tok_embd_, err)) return false;
    if (!find("output_norm.weight", out_norm_, err)) return false;
    hp_.n_vocab = (int32_t) tok_embd_.ne1;
    {
        tref sep;
        std::string ignore;
        if (find("output.weight", sep, ignore)) {
            err = "this model has a separate output head; only the tied form is handled";
            return false;
        }
    }

    L_.resize(hp_.n_layer);
    char name[128];
    for (int il = 0; il < hp_.n_layer; il++) {
        auto get = [&](const char * suffix, tref & out) {
            snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
            return find(name, out, err);
        };
        layer_refs & L = L_[il];
        if (!get("attn_norm.weight",   L.attn_norm)) return false;
        if (!get("attn_q.weight",      L.wq))        return false;
        if (!get("attn_k.weight",      L.wk))        return false;
        if (!get("attn_v.weight",      L.wv))        return false;
        if (!get("attn_output.weight", L.wo))        return false;
        if (!get("attn_q_norm.weight", L.q_norm))    return false;
        if (!get("attn_k_norm.weight", L.k_norm))    return false;
        if (!get("ffn_norm.weight",    L.ffn_norm))  return false;
        if (!get("ffn_gate.weight",    L.gate))      return false;
        if (!get("ffn_up.weight",      L.up))        return false;
        if (!get("ffn_down.weight",    L.down))      return false;

        for (const tref * t : { &L.wq, &L.wk, &L.wv, &L.wo, &L.gate, &L.up, &L.down }) {
            const ggml_type ty = (ggml_type) t->type;
            if (ty != GGML_TYPE_F32 && ty != GGML_TYPE_F16 && ty != GGML_TYPE_BF16) quantized_ = true;
        }
    }

    read_raw(tok_embd_, tok_embd_raw_);
    return true;
}

void qwen3_ref::read_raw(const tref & t, std::vector<uint8_t> & buf) const {
    buf.resize(t.size);
    FSEEK64(f_, (int64_t) (data_off_ + t.offset), SEEK_SET);
    if (fread(buf.data(), 1, t.size, f_) != t.size) {
        fprintf(stderr, "short read on tensor %lld\n", (long long) t.id);
        exit(1);
    }
}

void qwen3_ref::dequant_row(const tref & t, const uint8_t * raw, int64_t row, float * dst) const {
    const ggml_type type = (ggml_type) t.type;
    const size_t row_bytes = ggml_row_size(type, t.ne0);
    const uint8_t * src = raw + row * row_bytes;
    if (type == GGML_TYPE_F32) {
        memcpy(dst, src, t.ne0 * sizeof(float));
    } else {
        ggml_get_type_traits(type)->to_float(src, dst, t.ne0);
    }
}

std::vector<float> qwen3_ref::dequant(const tref & t) const {
    std::vector<uint8_t> raw;
    read_raw(t, raw);
    std::vector<float> out((size_t) t.ne0 * std::max<int64_t>(1, t.ne1));
    const int64_t rows = std::max<int64_t>(1, t.ne1);
    for (int64_t r = 0; r < rows; r++) {
        dequant_row(t, raw.data(), r, out.data() + r * t.ne0);
    }
    return out;
}

std::vector<float> qwen3_ref::norm_weights(int layer, norm which) const {
    const layer_refs & L = L_.at(layer);
    switch (which) {
        case norm::attn: return dequant(L.attn_norm);
        case norm::q:    return dequant(L.q_norm);
        case norm::k:    return dequant(L.k_norm);
        case norm::ffn:  return dequant(L.ffn_norm);
    }
    return {};
}

std::vector<float> qwen3_ref::projection_weights(int layer, unsigned which, int64_t & N, int64_t & K) const {
    const layer_refs & L = L_.at(layer);
    std::vector<const tref *> parts;
    switch (which) {
        case qwen3_prefill_opts::QKV:     parts = { &L.wq, &L.wk, &L.wv }; break;
        case qwen3_prefill_opts::O:       parts = { &L.wo }; break;
        case qwen3_prefill_opts::GATE_UP: parts = { &L.gate, &L.up }; break;
        case qwen3_prefill_opts::DOWN:    parts = { &L.down }; break;
        default: N = K = 0; return {};
    }
    std::vector<float> w;
    N = 0;
    K = parts[0]->ne0;
    for (const tref * t : parts) {
        const std::vector<float> part = dequant(*t);
        w.insert(w.end(), part.begin(), part.end());
        N += t->ne1;
    }
    return w;
}

bool qwen3_ref::prefill(const std::vector<int32_t> & toks, qwen3_prefill_out & out, int nth,
                        const qwen3_prefill_opts & opts) const {
    const auto t_start = std::chrono::steady_clock::now();
    const qwen3_rounding rr = opts.rounding;

    // a projection on the host with the configured rounding, or wherever
    // opts.projection sends it
    auto project = [&](int il, unsigned which, const float * xin, int64_t Tn, int64_t K, float * y, int64_t N) {
        if (opts.projection) return opts.projection(il, which, xin, Tn, K, y, N);
        int64_t wn, wk;
        std::vector<float> w = projection_weights(il, which, wn, wk);
        matmul_r(w, N, K, xin, Tn, y, nth, opts, which);
        if (opts.out_bf16 & which)
            for (int64_t i = 0; i < Tn * N; i++) y[i] = round_bf16(y[i]);
        return true;
    };
    const int64_t T  = toks.size();
    const int64_t E  = hp_.n_embd;
    const int64_t Hq = hp_.n_head, Hkv = hp_.n_head_kv, D = hp_.head_dim;
    const int64_t F  = hp_.n_ff;
    const int64_t Ek = Hkv * D;

    out.k.assign(hp_.n_layer, {});
    out.v.assign(hp_.n_layer, {});

    // residual stream
    std::vector<float> x((size_t) T * E);
    for (int64_t t = 0; t < T; t++) {
        dequant_row(tok_embd_, tok_embd_raw_.data(), toks[t], x.data() + t * E);
    }

    std::vector<float> h((size_t) T * E), q((size_t) T * Hq * D), k((size_t) T * Ek), v((size_t) T * Ek);
    std::vector<float> a((size_t) T * Hq * D), o((size_t) T * E);
    std::vector<float> g((size_t) T * F);

    for (int il = 0; il < hp_.n_layer; il++) {
        const layer_refs & L = L_[il];

        const std::vector<float> attn_norm = dequant(L.attn_norm);
        rmsnorm(x.data(), attn_norm.data(), T, E, hp_.rms_eps, h.data());

        {
            // one fused projection, split afterwards into q, k and v
            const int64_t Nqkv = Hq * D + 2 * Ek;
            std::vector<float> qkv((size_t) (T * Nqkv));
            if (!project(il, qwen3_prefill_opts::QKV, h.data(), T, E, qkv.data(), Nqkv)) return false;
            for (int64_t t = 0; t < T; t++) {
                const float * row = qkv.data() + t * Nqkv;
                std::copy(row,                row + Hq * D,          q.data() + t * Hq * D);
                std::copy(row + Hq * D,       row + Hq * D + Ek,     k.data() + t * Ek);
                std::copy(row + Hq * D + Ek,  row + Nqkv,            v.data() + t * Ek);
            }
        }

        // per-head rmsnorm on q and k, then rotary
        {
            const std::vector<float> qn = dequant(L.q_norm), kn = dequant(L.k_norm);
            rmsnorm(q.data(), qn.data(), T * Hq,  D, hp_.rms_eps, q.data());
            rmsnorm(k.data(), kn.data(), T * Hkv, D, hp_.rms_eps, k.data());
        }
        rope_neox(q.data(), T, Hq,  D, hp_.rope_base);
        rope_neox(k.data(), T, Hkv, D, hp_.rope_base);

        // what the cache holds for this layer
        out.k[il].resize((size_t) T * Ek);
        out.v[il].resize((size_t) T * Ek);
        ggml_fp32_to_fp16_row(k.data(), (ggml_fp16_t *) out.k[il].data(), T * Ek);
        ggml_fp32_to_fp16_row(v.data(), (ggml_fp16_t *) out.v[il].data(), T * Ek);

        if (opts.round_attention && rr != qwen3_rounding::none) {
            attention_r(q.data(), k.data(), v.data(), T, Hq, Hkv, D, a.data(), nth, rr, opts.attn_scores, opts.attn_values,
                        opts.attn_scores_split, opts.scores_bf16, &opts);
        } else {
            attention(q.data(), k.data(), v.data(), T, Hq, Hkv, D, a.data(), nth, opts.scores_bf16);
        }
        if (opts.attn_out_bf16)
            for (float & e : a) e = round_bf16(e);

        if (!project(il, qwen3_prefill_opts::O, a.data(), T, Hq * D, o.data(), E)) return false;
        for (size_t i = 0; i < x.size(); i++) x[i] += o[i];

        const std::vector<float> ffn_norm = dequant(L.ffn_norm);
        rmsnorm(x.data(), ffn_norm.data(), T, E, hp_.rms_eps, h.data());
        {
            std::vector<float> gu((size_t) (T * 2 * F));
            if (!project(il, qwen3_prefill_opts::GATE_UP, h.data(), T, E, gu.data(), 2 * F)) return false;
            for (int64_t t = 0; t < T; t++) {
                const float * row = gu.data() + t * 2 * F;
                for (int64_t i = 0; i < F; i++) {
                    const float gi = silu(row[i]) * row[F + i];
                    g[t * F + i] = opts.swiglu_bf16 ? round_bf16(gi) : gi;
                }
            }
        }
        if (!project(il, qwen3_prefill_opts::DOWN, g.data(), T, F, o.data(), E)) return false;
        for (size_t i = 0; i < x.size(); i++) x[i] += o[i];
    }

    // logits for the last token, through the tied embedding
    {
        const std::vector<float> on = dequant(out_norm_);
        std::vector<float> hn(E);
        rmsnorm(x.data() + (T - 1) * E, on.data(), 1, E, hp_.rms_eps, hn.data());
        out.logits_last.assign(hp_.n_vocab, 0.0f);
        parallel_for(hp_.n_vocab, nth, [&](int64_t b, int64_t e) {
            std::vector<float> row(E);
            for (int64_t r = b; r < e; r++) {
                dequant_row(tok_embd_, tok_embd_raw_.data(), r, row.data());
                out.logits_last[r] = dot(row.data(), hn.data(), E);
            }
        });
    }

    out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    return true;
}
