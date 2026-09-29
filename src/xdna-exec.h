// Block claiming (plan: .claude/plans/backend-size-aware.md, step 5): runs one
// graph piece the scheduler handed us, the matmuls on the NPU and the small
// ops between them on the host.
//
// Every op in a claimed piece works row by row (one row per prompt token), so
// the piece's rows are split into streams and the streams take turns: the host
// runs one stream's small ops while the NPU runs another's matmuls, as in the
// prototype (hybrid/npu_prefill.cpp). Between two matmuls the host makes one
// pass over the rows, a few at a time: it reads the NPU's output, runs the
// small ops, and encodes the next matmul's input. Results used only inside
// that pass never leave the cache; the rest stay in host memory, and only
// those something outside the piece reads go back to Vulkan.
#pragma once

#include "ggml.h"

#include <functional>
#include <string>

class xdna_npu;

// What the executor needs from the backend.
struct xdna_io {
    // a tensor's bytes, from wherever it lives, into host memory
    std::function<void(const ggml_tensor *, void *)> read;
    // host memory into a tensor
    std::function<void(ggml_tensor *, const void *)> write;
    // a weight-like leaf (a norm's scale), read once and kept
    std::function<const float *(const ggml_tensor *)> param;
    // host memory for the piece, freed when it ends; `pinned` memory is where
    // the GPU copies inputs fastest
    std::function<void *(size_t)> alloc;
    // whether a result must go back to Vulkan: something outside the piece
    // (or something we never saw) reads it
    std::function<bool(const ggml_tensor *, const ggml_cgraph *)> needed_outside;
    // an NPU matmul's weight: made ready (8-bit copy built) and identified
    std::function<bool(const ggml_tensor *, std::string &)> ensure_weight;
    // a gate and an up weight, fused for SiLU(gate) * up on the NPU
    std::function<bool(const ggml_tensor *, const ggml_tensor *, std::string &)> ensure_fused;
};

// Host time in those passes, and how many, since the last reset, for
// GGML_XDNA_TRACE.
extern double xdna_pass_ms;
extern int64_t xdna_passes;

// Whether the executor implements this op, in this variant. The policy asks
// this before claiming anything but a matmul.
bool xdna_exec_supports(const ggml_tensor * op);

// Runs the piece. Returns false with `err` set on failure.
bool xdna_exec_piece(const ggml_cgraph * g, xdna_npu & npu, const xdna_io & io, int n_streams, std::string & err);
