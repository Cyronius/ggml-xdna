// Qwen3 prefill with every projection and attention's two matmuls on the NPU
// (the bfp16 kernel, hybrid/npu_bfp16.h) and the rest on the host, fast
// enough to time against the GPU. Plan: .claude/plans/npu-prefill-beats-gpu.md,
// stage 3e.
//
// Layer-major: the whole prompt goes through one layer before the next. The
// prompt is split at 512-token boundaries into streams (two by default), and
// each stream goes through a layer as alternating host and NPU steps: norm,
// q/k/v, attention's scores and values block by block, output, gate/up, down.
// The streams take turns, so the host works on one stream while the NPU runs
// the other's step. The only dependency between them is that a later stream's
// attention reads an earlier stream's keys and values, and the earlier stream
// is always a step ahead.
//
// The host work that follows each projection (splitting q/k/v, the per-head
// norms and rotary, SiLU(gate) * up, the residual adds) is done while reading
// the NPU's output, in one pass, and whatever produces a projection's input
// (a norm, attention, SiLU(gate) * up) encodes it straight into the NPU's
// buffers as it goes.
//
// Produces what the GPU context needs and nothing else: every layer's keys and
// values in llama.cpp's cache layout (qwen3_prefill_out::k / ::v). No logits;
// the GPU runs the last prompt token itself and gets them there.
#pragma once

#include "bfp16_pack.h"
#include "gpu_attention.h"
#include "npu_bfp16.h"
#include "qwen3_ref.h"
#include "thread_pool.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct npu_prefill_stats {
    double total_ms = 0;
    double wait_ms = 0;   // the host with nothing to do but wait for the NPU (includes output syncs)
    double norm_ms = 0;   // the norms, encoding their output as a projection's input
    double read_ms = 0;   // reading projection outputs (q/k/v and rotary, residual adds, SiLU(gate) * up)
    double read_qkv_ms = 0, read_o_ms = 0, read_gu_ms = 0, read_down_ms = 0;  // read_ms by projection
    double attn_in_ms = 0, attn_soft_ms = 0, attn_out_ms = 0;  // attention's host work, by phase
    double prep_ms = 0;   // readying a projection's calls: buffers, and zeroing input rows past the prompt
    double start_ms = 0;  // starting NPU work: syncing its operands to the device, then submitting it
    double sync_ms = 0;   // buffer syncs to and from the NPU (inside the columns above)
    double sync_in_ms = 0, sync_out_ms = 0;  // sync_ms split: operands to the device, outputs back
    double other_ms = 0;  // embedding and rotary tables
    // attention on the GPU: the host waiting for it; and the GPU worker's own
    // time copying in, running, copying out and building graphs (not host time)
    double gpu_wait_ms = 0, gpu_in_ms = 0, gpu_run_ms = 0, gpu_out_ms = 0, gpu_build_ms = 0;
    bool   gpu_out_imported = false;  // the GPU wrote its output straight into cached host memory
    int    calls = 0;     // NPU calls
    int    streams = 0;
};

class npu_prefill {
public:
    // Rows per NPU projection call: 1024, or 512 for a short remainder (the
    // smallest the kernel's tiling allows). Attention goes 512 queries a block.
    static constexpr int64_t BLOCK = 512, BIG = 1024;
    static constexpr int64_t KV_ROWS = 4096;  // the longest prompt the shared key/value buffers hold

    // `build_root` holds the kernel builds as <M>x<K>x<N>_128x64x64_c8/, all
    // sharing one core program that takes K at run time (whole_array_bfp_rtp),
    // so every shape runs under the first build's xclbin in one hardware
    // context. A root whose name ends in "_ct" holds --c-tiled builds, whose
    // outputs are read through bfp16_c_row. Attention's shapes come from
    // <build_root>kv/ when it exists (built with --b-groups / --b-kfull
    // KV_ROWS): then each layer's keys and transposed values are encoded once,
    // into one buffer per kv head that every query block's calls read a
    // prefix of, instead of again for every block. With out16, a projection
    // whose root also has <M>x<K>x<N>_128x64x64_c8_m<mode>/ at both M (built
    // with --out-mode, whole_array_bfp_rtp's output modes) runs in that mode.
    bool load(const qwen3_ref & model, const std::string & build_root, int n_threads, std::string & err);
    bool prefill(const std::vector<int32_t> & toks, qwen3_prefill_out & out, npu_prefill_stats & st, std::string & err);

    // Where attention runs (plan: .claude/plans/attention-gpu-or-npu.md). npu:
    // its two products on the NPU, the softmax between them on the host, for
    // prompt lengths whose shapes are built (the host otherwise). gpu: all of
    // it on gpu_device, through ggml. cpu: all of it on the host in float32.
    enum class attn_where { npu, gpu, cpu };
    attn_where attn = attn_where::npu;
    ggml_backend_dev_t gpu_device = nullptr;  // for attn_where::gpu
    // How many streams the prompt is split into, at most. 1 runs every step
    // one after the other.
    int max_streams = 2;
    // Projections with 16-bit outputs, where the build root has them (see
    // load): q/k/v, o and down as bf16, gate/up as SiLU(gate) * up in bf16,
    // computed on the NPU. Off runs every projection in fp32, as before. Set
    // before load.
    bool out16 = true;
    // After load: whether any projection runs with 16-bit outputs.
    bool out16_used() const { return !L_.empty() && (L_[0].qkv.mode || L_[0].gate_up.mode); }

private:
    struct proj {
        int shape = -1, shape_big = -1, weights = -1;  // shape_big: BIG rows, -1 if not built
        int64_t K = 0, N = 0;
        int mode = 0;  // output mode: 0 fp32, 1 bf16, 2 SiLU(gate) * up in bf16 (N/2 values a row)
    };
    struct layer {
        std::vector<float> attn_norm, q_norm, k_norm, ffn_norm;
        proj qkv, o, gate_up, down;
    };
    // One NPU call's M x N output, row-major or --c-tiled, in a projection's
    // output mode.
    struct out_view {
        const float * y;
        int64_t M, N;
        const bfp16_tiling * tiled;  // null when row-major
        int mode;
        // Row r: in place when row-major fp32, else copied into `scratch` (N floats).
        const float * row(int64_t r, float * scratch) const {
            if (mode) bfp16_c16_row(y, M, N, *tiled, mode, r, scratch);
            else if (tiled) bfp16_c_row(y, M, N, *tiled, r, N, scratch);
            else return y + r * N;
            return scratch;
        }
    };
    // A run of the prompt's rows, [t0, t1), starting on a 512-query block.
    // Its projections go to the NPU as one call per BIG rows, each call with
    // its own input and output buffers; its attention as one batch of scores
    // and one of values per query block.
    struct stream {
        int64_t t0 = 0, t1 = 0;
        std::unique_ptr<npu_bfp16::batch> proj, sc, pv;
        int gpu_ticket = -1;  // its attention on the GPU, while that runs
        int calls() const { return (int) ((t1 - t0 + BIG - 1) / BIG); }
        int64_t call_rows(int j) const { return std::min(BIG, t1 - t0 - j * BIG); }
        int64_t call_m(int j) const { return call_rows(j) > BLOCK ? BIG : BLOCK; }
    };
    // Everything one prefill's steps share.
    struct pass;

    bool make_proj(const qwen3_ref & model, int il, unsigned which, proj & p, std::string & err);
    // Host step k of layer `il` for stream s, then its NPU work started. Sets
    // `npu` to what was started (null for none) and `done` after the last step.
    bool step(pass & P, stream & s, int il, int k, npu_bfp16::batch *& npu, bool & done, std::string & err);
    // The projection's calls for s's rows, inputs zeroed past the prompt.
    bool prepare_proj(pass & P, stream & s, const proj & p, std::string & err);
    // Row t (in s) of a projection's input, encoded into its call's buffer.
    void encode_row(const stream & s, int64_t K, int64_t t, const float * row) const;
    // Attention on the NPU for query block b, in three host steps around its
    // two NPU batches: the scores' operands (per kv head, [q | q | q - bfp16(q)]
    // against [k | k - bfp16(k) | k]), then the causal softmax into
    // probabilities times values' operands, then the result encoded as the
    // output projection's input.
    bool attn_scores(pass & P, stream & s, int64_t b, std::string & err);
    // s's rows of every kv head's keys and transposed values, into the shared
    // buffers (rows past the prompt as zeros, to the end of s's last block).
    bool encode_kv(pass & P, const stream & s, std::string & err);
    bool attn_softmax(pass & P, stream & s, int64_t b, std::string & err);
    void attn_output(pass & P, stream & s, int64_t b);
    // Attention on the host, float32, for rows [t0, t1).
    void attention_host(const float * q, const float * k, const float * v, int64_t t0, int64_t t1, float * a);
    bool attention_on_npu(int64_t T) const;

    std::map<int64_t, int> scores_, pv_;  // attention's shapes by key length L
    bool c_tiled_ = false;
    bool kv_once_ = false;
    std::vector<int> keys_, vt_;          // per kv head, with kv_once_: [k | k - bfp16(k) | k] rows; v transposed

    const qwen3_ref * model_ = nullptr;
    qwen3_hparams hp_;
    std::string root_;
    bfp16_tiling tile_;
    std::unique_ptr<thread_pool> pool_;
    std::unique_ptr<gpu_attention> gpu_;  // started on the first prefill with attn_where::gpu
    npu_bfp16 npu_;
    std::map<int64_t, int> shape_for_;  // by M, K, N
    std::vector<layer> L_;
    std::vector<stream> streams_;
    size_t a_bytes_ = 0, c_bytes_ = 0;  // a projection call's buffers: BIG rows at the largest K, N
};
