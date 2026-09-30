// The NPU's bfp16 matmul (kernels/bfp16_gemm, with float32 running sums)
// driven through the vendored XRT shim.
//
// One hardware context for everything. Switching contexts costs about 2.4 ms
// on this NPU against about 0.1 ms for a dispatch, so every shape runs as its
// own instruction stream under the one xclbin (one xclbin serves any M and N
// at the K it was built for). Weights are packed once and stay resident in
// their own buffers; each call packs nothing, it takes operands already in the
// kernel's layout (bfp16_pack.h).
#pragma once

#include "bfp16_pack.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

class npu_bfp16 {
public:
    ~npu_bfp16();
    bool open(const std::string & xclbin_path, std::string & err);
    void close();

    // One shape's instruction stream (a build directory's insts.bin). Returns
    // its handle, or -1.
    int add_shape(const std::string & insts_path, int64_t M, int64_t K, int64_t N, std::string & err);
    // The same from the stream's words (bfp16_insts.h).
    int add_shape(const std::vector<uint32_t> & insts, int64_t M, int64_t K, int64_t N, std::string & err);

    // A weight operand already in the kernel's layout (bfp16_pack_b), copied
    // into its own device buffer. Returns its handle, or -1.
    int add_weights(const std::vector<uint8_t> & packed, std::string & err);

    // A B operand the host rewrites in place (attention's keys and values):
    // `bytes` of zeros, filled through buffer_map and synced to the device a
    // range at a time. Its handle goes where add_weights' do.
    int add_buffer(size_t bytes, std::string & err);
    uint8_t * buffer_map(int buffer);
    bool buffer_sync(int buffer, size_t off, size_t n, std::string & err);

    // The activation buffer for the next call, sized for `shape`: pack into it
    // directly (bfp16_pack_a writes a std::vector; `set_a` copies one in).
    bool set_a(int shape, const std::vector<uint8_t> & packed, std::string & err);

    // C[M x N] = A * W^T for the operands set, float32 row-major into `c`.
    // `ms` is the submit-to-complete time.
    bool run(int shape, int weights, float * c, std::string & err, double * ms = nullptr);

    // The same without copies: pack A straight into a_map's buffer, then
    // run_mapped syncs it, runs, and points `c` at the mapped output (valid
    // until the next call). Activation buffers come in numbered slots, so a
    // caller can fill several calls' A before running any of them.
    uint8_t * a_map(int shape, std::string & err, int slot = 0);
    bool run_mapped(int shape, int weights, const float *& c, std::string & err, double * ms = nullptr, int slot = 0);
    // Slot `slot`'s activation buffer, at least `bytes` long. Asking for more
    // than it has reallocates it, losing what was there.
    uint8_t * a_slot(int slot, size_t bytes, std::string & err);

    const std::string & device_name() const { return device_name_; }

    // Time spent syncing buffers (cache flushes to and from the NPU), in
    // both run_mapped and batches: operands going to the NPU, outputs coming
    // back, and the two together. The caller resets them.
    double sync_in_ms = 0, sync_out_ms = 0, sync_ms = 0;

    // Several calls with their own A, B and C buffers, submitted together as
    // one XRT runlist: about 0.04 ms less per call than a start and wait each,
    // and the host fills every call's operands before and reads every output
    // after. Submitting and waiting are separate, so the host can work while
    // the NPU runs; the NPU runs submissions in order, so another batch can be
    // started behind this one. Buffers only grow; reserve sizes them up front
    // so that a(i) and c(i) never move.
    class batch {
    public:
        explicit batch(npu_bfp16 & npu) : npu_(npu) {}
        ~batch();
        bool reserve(int n, size_t a_bytes, size_t b_bytes, size_t c_bytes, std::string & err);
        // Call i runs shapes[i]. With `b` (add_weights or add_buffer handles,
        // one for every call or one per call), a call's B is that buffer
        // instead of its own; the caller keeps it synced.
        bool prepare(const std::vector<int> & shapes, std::string & err, const std::vector<int> & b = {});
        // Sync back only the first `bytes` of each output (0: all of it), when
        // the rest is never read.
        void sync_c_prefix(size_t bytes) { c_prefix_ = bytes; }
        uint8_t * a(int i) const { return calls_[i].a_map; }
        uint8_t * b(int i) const { return calls_[i].b_map; }
        // Syncs the operands and submits every call, without waiting.
        bool start(std::string & err);
        // Waits for what start submitted, then makes the outputs readable.
        // `ms`: start to done.
        bool wait(std::string & err, double * ms = nullptr);
        bool run(std::string & err, double * ms = nullptr) { return start(err) && wait(err, ms); }
        const float * c(int i) const { return (const float *) calls_[i].c_map; }

    private:
        struct call {
            void * a = nullptr, * b = nullptr, * c = nullptr, * run = nullptr;
            size_t a_cap = 0, b_cap = 0, c_cap = 0;
            uint8_t * a_map = nullptr, * b_map = nullptr, * c_map = nullptr;
            int shape = -1;
        };
        bool map(call & c, std::string & err);
        npu_bfp16 & npu_;
        std::vector<call> calls_;
        size_t n_ = 0;
        std::vector<int> b_;  // external B per call, empty when the batch owns them
        size_t c_prefix_ = 0;
        void * rl_ = nullptr;  // the runlist in flight, when more than one call
        bool in_flight_ = false;
        std::chrono::steady_clock::time_point t_start_;
    };

private:
    struct shape {
        int64_t M = 0, K = 0, N = 0;
        void *  ibo = nullptr;
        int     n_words = 0;
    };
    struct wbuf {
        void * bo = nullptr;
        size_t bytes = 0;
    };
    bool ensure(void *& bo, size_t & have, size_t need, std::string & err);
    bool submit(const shape & s, int slot, const wbuf & w, std::string & err, double * ms);

    void * dev_ = nullptr, * ctx_ = nullptr, * kern_ = nullptr, * run_ = nullptr;
    void * cbo_ = nullptr;
    size_t c_bytes_ = 0;
    std::vector<void *> abos_;  // activation buffers by slot
    std::vector<size_t> a_caps_;
    std::vector<shape> shapes_;
    std::vector<wbuf>  weights_;
    std::string device_name_;
};
