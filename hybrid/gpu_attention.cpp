#include "gpu_attention.h"

#include "ggml-alloc.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <malloc.h>
#include <vector>

namespace {
using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

// Key rows are padded to this, as llama.cpp's cache pads its view of them;
// the padding is masked out.
constexpr int64_t KV_PAD = 256;
// Host memory handed to the GPU has to start and end on the driver's import
// alignment; 64 KB covers every value drivers report.
constexpr size_t HOST_ALIGN = 65536;
}

gpu_attention::~gpu_attention() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    for (auto & [key, g] : graphs_) {
        ggml_gallocr_free(g.alloc);
        if (g.out_buf) ggml_backend_buffer_free(g.out_buf);
        if (g.out_host) _aligned_free(g.out_host);
        ggml_free(g.ctx);
    }
    if (backend_) ggml_backend_free(backend_);
}

bool gpu_attention::init(ggml_backend_dev_t dev, int64_t n_head, int64_t n_head_kv, int64_t head_dim, std::string & err) {
    if (!dev) { err = "no GPU device"; return false; }
    dev_ = dev;
    backend_ = ggml_backend_dev_init(dev, nullptr);
    if (!backend_) { err = "cannot start the GPU backend"; return false; }
    Hq_ = n_head;
    Hkv_ = n_head_kv;
    D_ = head_dim;
    worker_ = std::thread([this] { loop(); });
    return true;
}

gpu_attention::graph & gpu_attention::graph_for(int64_t t0, int64_t t1) {
    auto it = graphs_.find({ t0, t1 });
    if (it != graphs_.end()) return it->second;
    const clk::time_point h0 = clk::now();
    const int64_t rows = t1 - t0, nkv = (t1 + KV_PAD - 1) / KV_PAD * KV_PAD;
    graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    // q, k and v in the prefill's own row layout, [token][head][dim]; ggml's
    // attention takes them with tokens and heads swapped, as views
    g.q = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, D_, Hq_, rows);
    g.k = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, D_, Hkv_, nkv);
    g.v = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, D_, Hkv_, nkv);
    ggml_tensor * mask = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, nkv, rows);
    for (ggml_tensor * t : { g.q, g.k, g.v, mask }) ggml_set_input(t);
    ggml_tensor * qp = ggml_permute(g.ctx, g.q, 0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(g.ctx, g.k, 0, 2, 1, 3);
    ggml_tensor * vp = ggml_permute(g.ctx, g.v, 0, 2, 1, 3);
    // the result comes out as [token][head][dim], the prefill's layout again
    g.out = ggml_flash_attn_ext(g.ctx, qp, kp, vp, mask, 1.0f / std::sqrt((float) D_), 0.0f, 0.0f);
    ggml_prec_set_acc(g.out, GGML_PREC_F32);  // as llama.cpp asks for it
    ggml_set_output(g.out);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    {
        // the output in cached host memory, placed before the allocator runs
        // so it leaves the tensor alone
        const size_t size = (ggml_nbytes(g.out) + HOST_ALIGN - 1) / HOST_ALIGN * HOST_ALIGN;
        g.out_host = _aligned_malloc(size, HOST_ALIGN);
        g.out_buf = ggml_backend_dev_buffer_from_host_ptr(dev_, g.out_host, size, size);
        if (g.out_buf) {
            ggml_backend_tensor_alloc(g.out_buf, g.out, ggml_backend_buffer_get_base(g.out_buf));
        } else {
            _aligned_free(g.out_host);
            g.out_host = nullptr;
        }
        out_imported_ = g.out_buf != nullptr;
    }
    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    ggml_gallocr_alloc_graph(g.alloc, g.gf);

    // The mask depends only on the rows: query t0 + r sees keys 0 to t0 + r.
    // Key rows past t1 are zeros, set once here; start() copies only rows
    // below t1 after this.
    std::vector<ggml_fp16_t> m((size_t) (nkv * rows));
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int64_t r = 0; r < rows; r++)
        for (int64_t c = 0; c < nkv; c++) m[(size_t) (r * nkv + c)] = c <= t0 + r ? zero : ninf;
    ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_memset(g.k, 0, 0, ggml_nbytes(g.k));
    ggml_backend_tensor_memset(g.v, 0, 0, ggml_nbytes(g.v));
    {
        std::lock_guard<std::mutex> lk(mu_);
        times_.build_ms += ms_since(h0);
    }
    return graphs_.emplace(std::make_pair(t0, t1), g).first->second;
}

void gpu_attention::run(const job & j) {
    const int64_t Eq = Hq_ * D_, Ek = Hkv_ * D_, rows = j.t1 - j.t0;
    graph & g = graph_for(j.t0, j.t1);
    clk::time_point h0 = clk::now();
    ggml_backend_tensor_set(g.q, j.q + j.t0 * Eq, 0, (size_t) (rows * Eq) * sizeof(float));
    ggml_backend_tensor_set(g.k, j.k, 0, (size_t) (j.t1 * Ek) * sizeof(uint16_t));
    ggml_backend_tensor_set(g.v, j.v, 0, (size_t) (j.t1 * Ek) * sizeof(uint16_t));
    const double in = ms_since(h0);
    h0 = clk::now();
    const ggml_status st = ggml_backend_graph_compute(backend_, g.gf);
    ggml_backend_synchronize(backend_);
    const double run_ms = ms_since(h0);
    h0 = clk::now();
    if (g.out_host) {
        // One value through the backend's own read first: it makes the GPU's
        // writes visible to the host (a barrier) before the plain copy.
        float first;
        ggml_backend_tensor_get(g.out, &first, 0, sizeof(first));
        memcpy(j.out + j.t0 * Eq, g.out_host, (size_t) (rows * Eq) * sizeof(float));
    } else {
        ggml_backend_tensor_get(g.out, j.out + j.t0 * Eq, 0, (size_t) (rows * Eq) * sizeof(float));
    }
    const double out = ms_since(h0);
    std::lock_guard<std::mutex> lk(mu_);
    times_.in_ms += in;
    times_.run_ms += run_ms;
    times_.out_ms += out;
    if (st != GGML_STATUS_SUCCESS && error_.empty()) error_ = "GPU attention failed, status " + std::to_string((int) st);
}

void gpu_attention::loop() {
    for (;;) {
        job j;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !jobs_.empty(); });
            if (stop_) return;
            j = jobs_.front();
            jobs_.pop_front();
        }
        run(j);
        {
            std::lock_guard<std::mutex> lk(mu_);
            done_ = j.ticket;
        }
        cv_.notify_all();
    }
}

int gpu_attention::start(const float * q, const uint16_t * k, const uint16_t * v, int64_t t0, int64_t t1, float * out) {
    std::lock_guard<std::mutex> lk(mu_);
    const int ticket = next_++;
    jobs_.push_back({ ticket, q, k, v, t0, t1, out });
    cv_.notify_all();
    return ticket;
}

bool gpu_attention::wait(int ticket, std::string & err) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return done_ >= ticket; });
    if (!error_.empty()) { err = error_; return false; }
    return true;
}

gpu_attention::times gpu_attention::take_times() {
    std::lock_guard<std::mutex> lk(mu_);
    times t = times_;
    times_ = {};
    return t;
}
