// The NPU side of the backend: each claimed weight's private 8-bit copy, and
// the matmul itself, on the bfp16 kernel (npu/npu_bfp16.h, kernels/bfp16_gemm).
//
// The kernel is one core program (an xclbin of whole_array_bfp_rtp) that
// serves every size; each size's instruction stream is made here
// (npu/bfp16_insts.h).
// GGML_XDNA_KERNELS names the xclbin: the file itself, a directory holding
// final.xclbin, or a directory of per-size builds (any one's final.xclbin).
// A call covers 1024 prompt rows, or 512 for a short remainder.
#pragma once

#include "ggml.h"
#include "npu_bfp16.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

class thread_pool;

// The kernel's xclbin: GGML_XDNA_KERNELS if set (the file, or a directory
// holding final.xclbin or per-size builds), else bfp16_gemm.xclbin next to
// this DLL. Empty when there's none.
const std::string & xdna_xclbin();
// Whether the NPU can run the backend: it opens, loads the xclbin, and gets a
// small test multiply right. Checked once, when llama.cpp registers the
// backend; `why` says what failed.
bool xdna_npu_usable(std::string & why);
// Whether the kernel takes a weight of K x N at both row counts.
bool xdna_npu_has_shape(int64_t K, int64_t N);

class xdna_npu {
public:
    xdna_npu();
    ~xdna_npu();
    bool init(int n_threads, std::string & err);

    // Identifies a weight across calls: where it lives and what it is.
    struct wkey {
        const void * buffer, * data;
        int type;
        int64_t K, N;
        bool operator<(const wkey & o) const;
    };
    // Whether the weight's 8-bit copy exists already.
    bool has_weight(const wkey & key) const { return weights_.count(key) != 0; }
    // Builds it from the weight's raw ggml bytes: unpacked with ggml, encoded,
    // laid out for the kernel, and kept in its own NPU buffer. It runs in the
    // 16-bit output mode where that's built (half the bytes to read back;
    // GGML_XDNA_OUT16=0 keeps float32).
    bool add_weight(const wkey & key, const void * raw, std::string & err);

    // A gate and an up weight fused, for SiLU(gate) * up computed on the NPU
    // (the kernel's output mode 2): rows interleaved so each output tile pairs
    // a gate block with its up block. Its output has F = N values a row, the
    // SwiGLU result. Keyed by fused_key(gate). Needs the 2F-wide mode-2 builds.
    static wkey fused_key(const wkey & gate) { return { gate.buffer, gate.data, 1000 + gate.type, gate.K, 2 * gate.N }; }
    static bool has_fused_shape(int64_t K, int64_t F);
    bool add_fused(const wkey & gate, const void * gate_raw, const void * up_raw, std::string & err);

    // Output values a row for a weight added above (N, or N/2 fused).
    int64_t out_width(const wkey & key) const { return weights_.at(key).N_out; }

    // y[T x N] = x[T x K] * W^T, float32 row-major, for a weight added above.
    bool mul_mat(const wkey & key, const float * x, int64_t T, float * y, std::string & err);

    // The same in steps, for block claiming. `stream` picks a set of buffers,
    // with one group of calls in flight per stream: one call per weight, all
    // sharing one input (q, k and v; gate and up). The steps let the host read
    // one group's outputs and write the next group's input in the same pass
    // over the rows:
    //   reserve     sizes the stream's buffers for every group it will run,
    //               before the first, so none moves while outputs are unread
    //   begin       readies the next group, for `rows` rows (at most 1,024)
    //   encode_row  writes input row r (K values); any thread, one per row
    //   submit      starts the group
    //   wait        waits for it; its outputs can then be read until the
    //               next submit
    //   decode_row  reads row r of call k's output (out_width values); any
    //               thread. False if a value is NaN or infinite: then the
    //               NPU's results can't be trusted (not_finite_error says so)
    bool reserve(int stream, const std::vector<std::vector<wkey>> & groups, std::string & err);
    bool begin(int stream, const std::vector<wkey> & keys, int64_t rows, std::string & err);
    void encode_row(int stream, int64_t r, const float * row);
    bool submit(int stream, std::string & err);
    bool wait(int stream, std::string & err);
    bool decode_row(int stream, int k, int64_t r, float * row) const;
    static const char * not_finite_error();

    thread_pool & pool() { return *pool_; }

    // Time spent, since the last call: encoding activations, the NPU
    // (submit to done), reading its output, building weight copies.
    struct times { double encode_ms = 0, npu_ms = 0, decode_ms = 0, pack_ms = 0; int calls = 0; };
    times take_times();

private:
    struct weight {
        int handle = -1;
        int shape_small = -1, shape_big = -1;  // M = 512 and M = 1024
        int64_t K = 0, N = 0, N_out = 0;       // N as the kernel runs it (padded); N_out real values a row
        int mode = 0;                          // output mode: 0 float32, 1 bf16, 2 SiLU(gate) * up in bf16
    };
    int shape(int64_t M, int64_t K, int64_t N, int mode, std::string & err);
    bool add_packed(const wkey & key, std::vector<float> & f, int64_t N, int64_t n_out, int mode, std::string & err);
    void unpack(const void * raw, ggml_type type, int64_t N, int64_t K, float * dst);

    std::unique_ptr<npu_bfp16> npu_;
    std::unique_ptr<npu_bfp16::batch> batch_;  // one matmul's calls; buffers only grow
    struct flight {
        std::unique_ptr<npu_bfp16::batch> batch;
        std::vector<const weight *> w, w_out;  // the group being filled; the one whose outputs are readable
        int64_t rows = 0, M = 0, M_out = 0;
        std::vector<size_t> c_cap;             // output bytes reserved per call
        bool readable = false;
    };
    flight & flight_for(int stream);
    std::vector<flight> flights_;  // by stream
    std::unique_ptr<thread_pool> pool_;
    std::map<wkey, weight> weights_;
    std::map<std::tuple<int64_t, int64_t, int64_t, int>, int> shapes_;  // (M, K, N, mode) -> handle
    bool opened_ = false;
    times times_;
};
