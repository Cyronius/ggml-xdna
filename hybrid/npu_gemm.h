// Running one of the open prefill GEMM kernels on the NPU.
//
// The kernel computes Y[N,T] = W[N,K] @ X^T[K,T] with the weights streamed as
// 4-bit chunks (see q4_pack.h) and dequantised on the compute tiles, and the
// activations streamed as bfloat16 in a pre-tiled order the DMA walks
// directly. Both rearrangements are the host's job. This is the host side.
//
// A built kernel is a directory holding `final.xclbin` and `insts.bin`. The
// instruction stream fixes the shape, so one directory serves one (N, K, T).
//
// The tiling below is a port of openflowlm-next's `npu_offload/gemm_rtp/npue.py`
// `tile_b` at the settings this kernel family is built with. Its own tests
// check it against that file.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// The kernel's fixed tiling parameters, from gemm_q4_prefill.py.
constexpr int64_t NPU_GEMM_K_TILE = 64;
constexpr int64_t NPU_GEMM_TILE_N = 32;
constexpr int64_t NPU_GEMM_MAC_S  = 8;
constexpr int64_t NPU_GEMM_MAC_T  = 8;

// Rearrange activations for the kernel's DMA.
//
// `x_kt` is [K][T] row-major, one 16-bit element per value, which is the
// transpose of the natural [token][feature] layout. `out` takes K*T elements.
// Fails when the shape does not tile.
bool npu_gemm_tile_activations(const uint16_t * x_kt, int64_t K, int64_t T,
                               uint16_t * out, std::string & err);

// Holds a device, a context and one loaded kernel, with its buffers.
// Everything is allocated once; `run` is just a submit and wait.
class npu_gemm {
public:
    ~npu_gemm();

    // `build_dir` holds final.xclbin and insts.bin. The three buffer sizes are
    // weights, activations and output, in the order the kernel binds them.
    bool open(const std::string & build_dir, size_t w_bytes, size_t x_bytes, size_t y_bytes, std::string & err);

    // Copy into the device's weight and activation buffers.
    bool set_weights(const void * data, size_t n, std::string & err);
    bool set_activations(const void * data, size_t n, std::string & err);

    // Submit and wait. Returns the wall time in milliseconds through `ms`.
    bool run(double & ms, std::string & err);

    // Read the output back.
    bool get_output(void * data, size_t n, std::string & err);

    const std::string & device_name() const { return device_name_; }

private:
    void close();

    void * dev_ = nullptr, * ctx_ = nullptr, * kern_ = nullptr;
    void * ibo_ = nullptr, * wbo_ = nullptr, * xbo_ = nullptr, * ybo_ = nullptr;
    void * run_ = nullptr;
    int    n_instr_words_ = 0;
    std::string device_name_;
};
