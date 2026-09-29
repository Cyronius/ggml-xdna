// Every projection of a Qwen3 model on the NPU through the bfp16 kernel, as a
// qwen3_prefill_opts::projection. Plan: .claude/plans/npu-prefill-beats-gpu.md,
// stage 3c.
//
// All weights are packed once and stay resident. A call runs the prompt in
// blocks of 512 tokens (the last one zero-padded), one dispatch per block.
// The kernel builds live under `build_root` as <512>x<K>x<N>_128x64x64_c8/
// (final.xclbin, insts.bin), float32 running sums and float32 output.
//
// Until the kernel takes K at run time, each K (2048 and 6144 on Qwen3-1.7B)
// needs its own xclbin and so its own hardware context, and every change of K
// costs a context switch (about 2.4 ms). Fine for checking correctness; the
// timed driver waits for the single-xclbin build.
#pragma once

#include "bfp16_pack.h"
#include "npu_bfp16.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class qwen3_ref;

class npu_projections {
public:
    static constexpr int64_t BLOCK = 512;

    bool init(const qwen3_ref & model, const std::string & build_root, int n_threads, std::string & err);

    // qwen3_prefill_opts::projection
    bool operator()(int layer, unsigned which, const float * x, int64_t T, int64_t K, float * y, int64_t N);

    const std::string & error() const { return err_; }
    double npu_ms = 0, host_ms = 0;  // kernel time, and packing plus copies
    int    dispatches = 0;

private:
    struct context {
        int64_t K = 0;
        npu_bfp16 npu;
        std::map<int64_t, int> shape_for_n;
    };
    struct resident { context * ctx = nullptr; int weights = -1; };

    context * context_for(int64_t K, int64_t N, std::string & err);

    std::string root_;
    int nth_ = 1;
    bfp16_tiling tile_;
    std::vector<std::unique_ptr<context>> ctxs_;
    std::map<std::pair<int, unsigned>, resident> w_;
    std::vector<float> xblk_, yblk_;
    std::vector<uint8_t> apk_;
    std::string err_;
};
