// A plain host implementation of Qwen3's prefill, reading weights straight
// from a GGUF file and producing exactly what the hybrid design needs from the
// NPU: every layer's keys (after rotary encoding) and values for every prompt
// token, in the layout llama.cpp's cache stores them, plus the logits for the
// last token so the whole pass can be checked.
//
// It is deliberately simple and slow: float32 everywhere, weights dequantised
// on the fly, straightforward loops. Its job is to be an oracle. The NPU
// kernel is right when it agrees with this; until the kernel exists, this
// stands in for it so the rest of the pipeline can be built and tested.
//
// Qwen3 as llama.cpp runs it (src/models/qwen3.cpp in the pinned release):
//   per layer: rmsnorm -> q,k,v projections -> rmsnorm on q and k per head
//              -> rotary (the "neox" half-split form) on q and k
//              -> causal attention, grouped query, scale 1/sqrt(head_dim)
//              -> output projection -> residual
//              -> rmsnorm -> silu(gate) * up -> down -> residual
//   then rmsnorm -> output projection (tied to the token embedding here)
#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;

struct qwen3_hparams {
    int32_t n_embd = 0, n_layer = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_ff = 0, n_vocab = 0;
    float   rms_eps = 0, rope_base = 0;
};

// How the projection matmuls round their inputs. This stands in for an NPU
// datapath, so its effect on the whole model can be measured before the kernel
// exists. The norms, rotary and softmax stay in float32 in every mode.
enum class qwen3_rounding {
    none,         // float32 throughout: the oracle
    bf16,         // weights and activations to bfloat16, as today's open NPU kernels do
    q8_0,         // activations to 8-bit in blocks of 32 with a float16 scale, as llama.cpp's CPU path does on a 4-bit file
    bfp16,        // weights and activations to bfp16ebs8, rounded to nearest (conversion done on the host)
    bfp16_floor,  // weights to nearest when loaded, activations rounded down (conversion done on an AIE2P core)
};

const char * qwen3_rounding_name(qwen3_rounding r);

struct softmax_bias_stats;

struct qwen3_prefill_opts {
    qwen3_rounding rounding = qwen3_rounding::none;
    // Also round attention's two products the same way: queries and keys for
    // the scores, probabilities and values for the output. Off means attention
    // stays in float32 whatever the projections do.
    bool round_attention = false;
    // With round_attention: which of the two products to round. Scores are
    // queries times keys; the output is probabilities times values.
    bool attn_scores = true, attn_values = true;
    // Scores as a hi/lo split: q = qh + ql and k = kh + kl, each part rounded
    // on its own, and the scores taken as qh.kh + qh.kl + ql.kh. On the NPU
    // that is one bfp16 matmul three times as deep ([qh qh ql] against
    // [kh kl kh] along the head dimension).
    bool attn_scores_split = false;
    // For finding where the error comes from: leave one side unrounded, or
    // keep some projections in bf16 while the rest use `rounding`.
    bool keep_weights = false, keep_acts = false;
    enum : unsigned { QKV = 1, O = 2, GATE_UP = 4, DOWN = 8 };
    unsigned bf16_projections = 0;
    // How the kernel holds its running sums. An NPU matmul adds one chunk of
    // K at a time into an output tile it keeps in local memory. If that tile
    // is narrower than float32, the running sum is rounded (to nearest) after
    // every chunk of `acc_chunk` along K.
    enum class acc_type { f32, bf16, bfp16 } acc = acc_type::f32;
    int acc_chunk = 64;

    // Outputs the NPU hands back in bf16 instead of float32, each rounded to
    // nearest after the float32 sums are done (plan: .claude/plans/
    // smaller-npu-outputs.md). out_bf16: these projections' outputs (QKV, O,
    // GATE_UP, DOWN bits). swiglu_bf16: SiLU(gate) * up computed in float32
    // from gate/up's sums, then rounded, as a core that does SiLU would.
    // scores_bf16 / attn_out_bf16: attention's scores (before the scale and
    // softmax) and its output (probabilities times values).
    unsigned out_bf16 = 0;
    bool swiglu_bf16 = false, scores_bf16 = false, attn_out_bf16 = false;

    // The softmax as the NPU would do it (plan: .claude/plans/softmax-on-npu.md),
    // with round_attention and bfp16. q is scaled by log2(e)/sqrt(head dim)
    // before it is rounded, so scores come out in log2 units. Each row's bias b
    // is the midpoint of an upper bound on its largest score (the smaller of
    // |q| max|k| and the per-dimension bound, over the keys it sees) and a
    // lower bound (its scores against the first key, itself and the one
    // before), used when they are at most bias_window apart. Then p = 2^(s - b)
    // unnormalised, rounded to bfp16 along positions, and the output is p times
    // v divided by the sum of the rounded p. Rows whose bounds are further
    // apart take the normal path. exp_bf16 rounds p to bf16 first, standing in
    // for a cheap exp on the core.
    bool softmax_npu = false, exp_bf16 = false;
    float bias_window = 120.0f;
    softmax_bias_stats * bias_stats = nullptr;  // filled when set; not thread-safe across prefills

    // Runs a projection somewhere else (the NPU) instead of on the host:
    // y[T x N] = x[T x K] times the projection's weights transposed, for
    // `layer`'s projection `which` (QKV, O, GATE_UP or DOWN, fused as
    // projection_weights returns them). The rounding options above then do
    // not apply to that projection. Returning false stops the prefill.
    std::function<bool(int layer, unsigned which, const float * x, int64_t T, int64_t K, float * y, int64_t N)> projection;
};

// What the softmax_npu bounds look like on a real prompt (plan:
// .claude/plans/softmax-on-npu.md, step 0). Scores are in log2 units, the
// scale folded in. "max" is the true largest visible score of a query row.
struct softmax_bias_stats {
    int64_t rows = 0, rows_fallback = 0;    // (query, head) rows
    int64_t tiles = 0, tiles_fallback = 0;  // 128 queries of one head: one core's tile
    int64_t calls = 0, calls_fallback = 0;  // 512 queries of one kv head's heads: one scores call
    int64_t cs_tighter = 0;                 // rows where |q| max|k| beat the per-dimension bound
    // one bucket per log2 unit, the last one open
    static constexpr int NB = 512;
    std::vector<int64_t> up_gap = std::vector<int64_t>(NB), lo_gap = std::vector<int64_t>(NB),
                         window = std::vector<int64_t>(NB);
    double worst_up = 0, worst_lo = 0, worst_window = 0;
    void merge(const softmax_bias_stats & o);
};

struct qwen3_prefill_out {
    // [layer][token * n_head_kv * head_dim] as half floats: one cache row per
    // token, kv heads side by side, the way llama.cpp lays out a cell.
    std::vector<std::vector<uint16_t>> k, v;
    std::vector<float> logits_last;  // n_vocab
    double ms = 0;
};

class qwen3_ref {
public:
    ~qwen3_ref();
    bool load(const std::string & gguf_path, std::string & err);
    const qwen3_hparams & hp() const { return hp_; }
    // True when any layer weight is stored quantised. llama.cpp's CPU path
    // then rounds activations to 8-bit in blocks of 32 before every matmul,
    // which makes its keys and values a noisy yardstick on outlier tokens.
    bool quantized() const { return quantized_; }
    // Returns false only when opts.projection did.
    bool prefill(const std::vector<int32_t> & toks, qwen3_prefill_out & out, int n_threads,
                 const qwen3_prefill_opts & opts = {}) const;

    // One layer's projection weights as float32, N rows of K, fused the way
    // the NPU runs them: QKV is q's rows, then k's, then v's; GATE_UP is
    // gate's rows, then up's.
    std::vector<float> projection_weights(int layer, unsigned which, int64_t & N, int64_t & K) const;

    // The small per-layer vectors, as float32, for a driver that runs the
    // layer itself (hybrid/npu_prefill.cpp).
    enum class norm { attn, q, k, ffn };
    std::vector<float> norm_weights(int layer, norm which) const;
    std::vector<float> output_norm() const { return dequant(out_norm_); }
    // One token's embedding row, n_embd floats.
    void embedding(int32_t token, float * dst) const { dequant_row(tok_embd_, tok_embd_raw_.data(), token, dst); }

private:
    struct tref {
        int64_t id = -1;
        int     type = 0;
        size_t  offset = 0, size = 0;
        int64_t ne0 = 0, ne1 = 0;
    };
    struct layer_refs {
        tref attn_norm, wq, wk, wv, wo, q_norm, k_norm, ffn_norm, gate, up, down;
    };

    bool find(const char * name, tref & out, std::string & err) const;
    void read_raw(const tref & t, std::vector<uint8_t> & buf) const;
    // whole tensor as float32, rows of ne0
    std::vector<float> dequant(const tref & t) const;
    // one row of an already-read raw tensor
    void dequant_row(const tref & t, const uint8_t * raw, int64_t row, float * dst) const;

    qwen3_hparams hp_;
    bool           quantized_ = false;
    gguf_context * gguf_ = nullptr;
    ggml_context * gctx_ = nullptr;
    FILE *         f_    = nullptr;
    size_t         data_off_ = 0;

    std::vector<layer_refs> L_;
    tref tok_embd_, out_norm_;
    std::vector<uint8_t> tok_embd_raw_;  // kept: used for input lookup and the tied output head
};
