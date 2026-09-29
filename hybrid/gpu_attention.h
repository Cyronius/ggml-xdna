// Causal attention on the GPU through ggml's backend interface, for
// npu_prefill's attention switch (plan: .claude/plans/attention-gpu-or-npu.md).
//
// One ggml graph per (first query row, last row): q, k and v in, ggml's
// flash attention, the output back. A prefill's streams keep the same rows
// in every layer, so each graph is built once and reused for all of them.
// The work runs on a worker thread, in the order it was started, so the
// caller carries on (reading NPU output, starting NPU work) while the GPU
// runs, whether or not the backend's own compute call waits for the GPU.
#pragma once

#include "ggml-backend.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

class gpu_attention {
public:
    ~gpu_attention();
    bool init(ggml_backend_dev_t dev, int64_t n_head, int64_t n_head_kv, int64_t head_dim, std::string & err);

    // Starts attention for query rows [t0, t1) against keys [0, t1).
    // q: float32 rows of n_head * head_dim, indexed from row 0. k, v: float16
    // rows of n_head_kv * head_dim, from row 0. out: float32 rows like q; rows
    // t0 to t1 are written. Everything must stay put until wait() returns.
    // Returns a ticket for wait().
    int start(const float * q, const uint16_t * k, const uint16_t * v, int64_t t0, int64_t t1, float * out);
    bool wait(int ticket, std::string & err);

    // Time the worker spent, since the last reset: copying q, k and v in,
    // running the graph, copying the output back, and building graphs.
    struct times { double in_ms = 0, run_ms = 0, out_ms = 0, build_ms = 0; };
    times take_times();
    // Whether the output goes straight into cached host memory (see graph).
    bool out_imported() const { return out_imported_; }

private:
    struct graph {
        struct ggml_context * ctx = nullptr;
        struct ggml_cgraph *  gf  = nullptr;
        ggml_gallocr_t        alloc = nullptr;
        struct ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr;
        // The output's own memory: ordinary, cached host memory the GPU writes
        // into directly. Null when the driver won't import it, and then the
        // output lives in GPU memory the CPU reads uncached (slowly).
        void * out_host = nullptr;
        ggml_backend_buffer_t out_buf = nullptr;
    };
    struct job {
        int ticket;
        const float * q;
        const uint16_t * k, * v;
        int64_t t0, t1;
        float * out;
    };

    graph & graph_for(int64_t t0, int64_t t1);
    void run(const job & j);
    void loop();

    ggml_backend_dev_t dev_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int64_t Hq_ = 0, Hkv_ = 0, D_ = 0;
    std::map<std::pair<int64_t, int64_t>, graph> graphs_;

    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<job> jobs_;
    int next_ = 0, done_ = -1;
    bool stop_ = false;
    bool out_imported_ = false;
    std::string error_;
    times times_;
};
