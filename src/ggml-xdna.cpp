// ggml backend for the AMD XDNA2 NPU, working next to the Vulkan GPU.
//
// The device registers as an integrated GPU named XDNA0 and is opted in with
// `-dev XDNA0,Vulkan0`: listed first, it has the highest priority. It reports
// no memory of its own, so llama.cpp gives every layer's weights to Vulkan.
// Its buffer type *is* Vulkan's, so it can read every weight and activation
// where they already are, and the scheduler inserts no copies between the two
// backends. That makes supports_op the whole policy: the scheduler hands us
// any op we accept, and Vulkan runs everything else exactly as it would alone.
// supports_op accepts big prompt work and declines the rest (reply generation,
// short prompts).
//
// Without -dev naming XDNA0, llama.cpp keeps only the first integrated GPU it
// finds (Vulkan's), and runs as if this backend weren't there.
//
// Tensors in Vulkan's buffers have no CPU pointer (their `data` is a device
// offset). We read them with an async get through a Vulkan backend of our own,
// into pinned host memory: the GPU does the copy, ~20 GB/s on this machine,
// where the plain get copies out of memory the CPU doesn't cache at 0.2 GB/s.
// We write results with the plain set, a copy into mapped memory, which is
// fast.

#include "ggml-xdna.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "xdna-env.h"
#include "xdna-ref.h"
#ifdef XDNA_HAVE_NPU
#include "xdna-npu.h"
#include "xdna-exec.h"
#include "thread_pool.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#define XDNA_DESCRIPTION "AMD XDNA2 NPU (prompt work, next to the Vulkan GPU)"

// dst->ne[1] tokens for a plain matmul - the convention the Vulkan backend
// uses for its offload threshold.
static int64_t xdna_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT: return op->ne[1];
        default:              return ggml_nrows(op);
    }
}

static int64_t env_int(const char * name, int64_t def) { return xdna_env_int(name, def); }

// Batch size at or above which a matmul is ours: llama.cpp's default prompt
// chunk (-ub 512). There the NPU ties the GPU (0.98-1.07x, idle machine,
// 2026-09-30), and ties go to the NPU; at -ub 2048 it leads by 13-18%.
// The kernel's smallest call is 512 rows, so smaller chunks would be mostly
// padding.
static int64_t xdna_min_batch() {
    static int64_t v = env_int("GGML_XDNA_MIN_BATCH", 512);
    return v;
}

// A dispatch costs something whatever it computes, so tiny matmuls stay on
// the GPU even at a large batch.
static int64_t xdna_min_mflop() {
    static int64_t v = env_int("GGML_XDNA_MIN_MFLOP", 256);
    return v;
}

static int xdna_n_threads() {
    static int v = (int) env_int("GGML_XDNA_N_THREADS", [] {
        const unsigned hc = std::thread::hardware_concurrency();
        return hc ? (int) hc : 4;
    }());
    return v;
}

// How the backend runs, decided once, when llama.cpp registers it:
//   npu   the NPU opens and runs the kernel (xdna_npu_usable)
//   host  GGML_XDNA_HOST_ONLY=1: claimed matmuls run on a CPU reference,
//         for tests and CI on machines without an NPU
//   off   neither: no device is offered, and llama.cpp runs as if the backend
//         weren't there
enum class xdna_mode { off, npu, host };

static xdna_mode xdna_run_mode() {
    static const xdna_mode m = [] {
        if (env_int("GGML_XDNA_HOST_ONLY", 0) != 0) {
            GGML_LOG_INFO("xdna: host-only mode (GGML_XDNA_HOST_ONLY): claimed matmuls run on the CPU\n");
            return xdna_mode::host;
        }
#ifdef XDNA_HAVE_NPU
        std::string why;
        if (xdna_npu_usable(why)) {
            GGML_LOG_INFO("xdna: %s\n", why.c_str());
            return xdna_mode::npu;
        }
        GGML_LOG_WARN("xdna: not offering " GGML_XDNA_DEVICE_NAME ": %s\n", why.c_str());
#else
        GGML_LOG_WARN("xdna: not offering " GGML_XDNA_DEVICE_NAME ": built without the NPU\n");
#endif
        return xdna_mode::off;
    }();
    return m;
}

// Set when the NPU fails during a prompt. The piece that failed is finished
// on the CPU, and from then on the backend claims nothing, so everything goes
// to the GPU. Never set in host-only mode.
static std::atomic<bool> g_npu_broken{ false };

static void xdna_npu_failed(const std::string & err) {
    if (!g_npu_broken.exchange(true))
        GGML_LOG_ERROR("xdna: the NPU failed (%s); finishing this step on the CPU and handing everything to the GPU "
                       "from now on\n",
                       err.c_str());
}

//
// the Vulkan device we work next to
//

struct xdna_vk {
    ggml_backend_dev_t         dev       = nullptr;
    ggml_backend_buffer_type_t buft      = nullptr;  // its device memory: ours too
    ggml_backend_buffer_type_t host_buft = nullptr;  // its pinned host memory
};

// Looked up on first use rather than at registration: our DLL loads after
// Vulkan's, but the registry is still registering us while we'd look.
static const xdna_vk & xdna_vulkan() {
    static xdna_vk vk;
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_backend_dev_t d = ggml_backend_dev_by_name("Vulkan0");
        if (!d) {
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t c = ggml_backend_dev_get(i);
                const enum ggml_backend_dev_type t = ggml_backend_dev_type(c);
                if ((t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
                    strcmp(ggml_backend_dev_name(c), GGML_XDNA_DEVICE_NAME) != 0) {
                    d = c;
                    break;
                }
            }
        }
        if (!d) {
            GGML_LOG_ERROR("%s: no Vulkan device registered; " GGML_XDNA_DEVICE_NAME " works next to one\n", __func__);
            return;
        }
        vk.dev = d;
        vk.buft = ggml_backend_dev_buffer_type(d);
        vk.host_buft = ggml_backend_dev_host_buffer_type(d);
    });
    return vk;
}

//
// backend
//

// Host memory we read Vulkan's tensors into is Vulkan's pinned memory, where
// the GPU copies fastest. GGML_XDNA_PINNED=0 uses ordinary memory instead: a
// diagnostic, and for software Vulkan devices (CI's lavapipe on Windows hands
// out pinned memory ggml then rejects as misaligned).
static bool xdna_pinned_mem() {
    static const bool v = env_int("GGML_XDNA_PINNED", 1) != 0;
    return v;
}

struct xdna_context {
    ggml_backend_t vk = nullptr;  // our own Vulkan backend, for async reads

    // host memory the GPU copies our inputs into; grows as needed
    ggml_backend_buffer_t pin = nullptr;
    size_t pin_size = 0;

    // Each claimed weight's raw bytes, read once, for matmuls run on the
    // host (the NPU keeps its own 8-bit copy). Only tensors in buffers
    // llama.cpp marks as weights are kept (their contents never change after loading),
    // keyed on where they live and what they are rather than on the tensor
    // struct, whose address can be reused.
    struct wkey {
        const void * buffer, * data;
        int type;
        int64_t ne0, ne1;
        bool operator==(const wkey & o) const {
            return buffer == o.buffer && data == o.data && type == o.type && ne0 == o.ne0 && ne1 == o.ne1;
        }
    };
    struct wkey_hash {
        size_t operator()(const wkey & k) const {
            return std::hash<const void *>()(k.buffer) ^ (std::hash<const void *>()(k.data) * 31) ^ (size_t) k.ne1;
        }
    };
    std::unordered_map<wkey, std::vector<uint8_t>, wkey_hash> weights;
    std::vector<uint8_t> scratch_weight;  // a weight outside a weight buffer, read fresh each time

    std::vector<uint8_t> out;  // a result, before it goes back to Vulkan

    // where the time goes, printed with GGML_XDNA_TRACE=1 every `trace_every`
    // matmuls: reading inputs from Vulkan, writing results back, everything
    // inside graph_compute
    double read_ms = 0, write_ms = 0, compute_ms = 0;
    int64_t matmuls = 0, pieces = 0;

#ifdef XDNA_HAVE_NPU
    // the NPU, started on first use; null in host-only mode or once it failed
    std::unique_ptr<xdna_npu> npu;
    xdna_npu * get_npu() {
        if (g_npu_broken || xdna_run_mode() != xdna_mode::npu) return nullptr;
        if (npu) return npu.get();
        npu = std::make_unique<xdna_npu>();
        std::string err;
        if (!npu->init(xdna_n_threads(), err)) {
            xdna_npu_failed(err);
            npu.reset();
        }
        return npu.get();
    }
    // threads for pieces run on the host when the NPU has failed
    std::unique_ptr<thread_pool> host_pool;
    thread_pool & host_threads() {
        if (!host_pool) host_pool = std::make_unique<thread_pool>(xdna_n_threads());
        return *host_pool;
    }
#endif

    ~xdna_context() {
        for (pin_buf & p : pool) {
            if (p.buf) ggml_backend_buffer_free(p.buf);
            else ggml_aligned_free(p.mem, p.size);
        }
        if (pin) ggml_backend_buffer_free(pin);
        if (vk) ggml_backend_free(vk);
    }

    void * pinned(size_t size) {
        if (size > pin_size) {
            if (pin) ggml_backend_buffer_free(pin);
            pin = ggml_backend_buft_alloc_buffer(xdna_pinned_mem() ? xdna_vulkan().host_buft : ggml_backend_cpu_buffer_type(),
                                                 size);
            pin_size = pin ? size : 0;
        }
        return pin ? ggml_backend_buffer_get_base(pin) : nullptr;
    }

    // a tensor from Vulkan memory into host memory, the GPU doing the copy
    void read(const ggml_tensor * t, void * dst) {
        if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {  // a graph input the CPU holds
            memcpy(dst, t->data, ggml_nbytes(t));
            return;
        }
        ggml_backend_tensor_get_async(vk, t, dst, 0, ggml_nbytes(t));
        ggml_backend_synchronize(vk);
    }

    // Block claiming's host memory: pinned buffers (xdna_pinned_mem), handed
    // out per piece and all returned when it ends, reused by the next.
    struct pin_buf { ggml_backend_buffer_t buf; void * mem; size_t size; bool used; };
    std::vector<pin_buf> pool;
    void * pool_alloc(size_t n) {
        const bool pinned_mem = xdna_pinned_mem();
        pin_buf * best = nullptr;
        for (pin_buf & p : pool)
            if (!p.used && p.size >= n && (!best || p.size < best->size)) best = &p;
        if (!best) {
            pin_buf nb = { nullptr, nullptr, n, false };
            if (pinned_mem) {
                nb.buf = ggml_backend_buft_alloc_buffer(xdna_vulkan().host_buft, n);
                if (!nb.buf) return nullptr;
                nb.mem = ggml_backend_buffer_get_base(nb.buf);
            } else {
                nb.mem = ggml_aligned_malloc(n);
                if (!nb.mem) return nullptr;
            }
            pool.push_back(nb);
            best = &pool.back();
        }
        best->used = true;
        return best->mem;
    }
    void pool_release() { for (pin_buf & p : pool) p.used = false; }

    // small weight-like leaves (norm scales), read once
    std::unordered_map<wkey, std::vector<uint8_t>, wkey_hash> params;
    const float * param(const ggml_tensor * t) {
        const wkey key = { t->buffer, t->data, (int) t->type, t->ne[0], t->ne[1] };
        auto it = params.find(key);
        if (it != params.end()) return (const float *) it->second.data();
        std::vector<uint8_t> & v = params[key];
        v.resize(ggml_nbytes(t));
        read(t, v.data());
        return (const float *) v.data();
    }

    const std::vector<uint8_t> & weight(const ggml_tensor * w) {
        if (!w->buffer || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            scratch_weight.resize(ggml_nbytes(w));
            read(w, scratch_weight.data());
            return scratch_weight;
        }
        const wkey key = { w->buffer, w->data, (int) w->type, w->ne[0], w->ne[1] };
        auto it = weights.find(key);
        if (it != weights.end()) return it->second;
        std::vector<uint8_t> & bytes = weights[key];
        bytes.resize(ggml_nbytes(w));
        read(w, bytes.data());
        return bytes;
    }
};

//
// the memory budget for the NPU's weight copies
//

// What the NPU's copy of a K x N weight takes: 9 bytes for every 8 values,
// with N padded as the kernel runs it (pad_n in xdna-npu.cpp). A fused
// gate/up pair takes no more than its two weights apart. Host-only mode
// budgets the same, so it decides as the NPU would.
static double xdna_copy_bytes(int64_t K, int64_t N) {
    return (double) ((N + 511) / 512 * 512) * (double) K / 8 * 9;
}

// The layer a weight belongs to, from llama.cpp's names ("blk.12.attn_q.weight");
// -1 for the others (the output matrix).
static int xdna_layer_of(const ggml_tensor * w) {
    int layer = -1;
    return sscanf(w->name, "blk.%d.", &layer) == 1 ? layer : -1;
}

static constexpr double GB = 1024.0 * 1024 * 1024;

// The copies are built the first time each weight is used and freed with the
// context, so without a limit they could take more memory than the machine
// has. Weights are taken in the order llama.cpp asks about them (layer by
// layer, while it plans the first prompt at load) until the next would pass
// the limit; from then on that layer and every new weight stay on the GPU.
// What's taken stays taken, so the split never changes between prompts. When
// the last context goes, its copies go with it, and the next starts afresh.
//
// The limit is the memory free when the first weight is asked about (the
// model and its context are loaded by then), less what's kept back for
// llama.cpp's work buffers and the rest of the machine: the larger of 4 GB and
// a tenth of memory. GGML_XDNA_MAX_COPY_GB sets it instead.
//
// supports_op is asked with no context, so there's one budget per process:
// two contexts alive at once on the same model would each build the copies it
// counted once.
struct xdna_budget {
    std::mutex mu;
    int contexts = 0;  // backends alive
    bool started = false, cut = false, running = false, overridden = false;
    double limit = 0, free = 0, kept_back = 0, taken = 0;
    struct entry { int layer; double bytes; };
    std::unordered_map<xdna_context::wkey, entry, xdna_context::wkey_hash> in, out;  // taken; left to the GPU

    // where the limit came from, for the log
    std::string source() const {
        char s[96];
        if (overridden) snprintf(s, sizeof(s), "set by GGML_XDNA_MAX_COPY_GB");
        else snprintf(s, sizeof(s), "%.1f GB free, less %.1f GB kept back", free / GB, kept_back / GB);
        return s;
    }

    void reset() {
        started = cut = running = overridden = false;
        limit = free = kept_back = taken = 0;
        in.clear();
        out.clear();
    }
};
static xdna_budget g_budget;

// Whether the weight's copy fits the budget. Weights with no data (llama.cpp's
// memory-fitting trial and load-time checks), weights outside a weight buffer
// (nothing is kept for them) and questions asked with no context alive build
// nothing, so they pass.
static bool xdna_budget_takes(const ggml_tensor * w) {
    if (!w->buffer || !w->data || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
        return true;
    xdna_budget & b = g_budget;
    std::lock_guard<std::mutex> lk(b.mu);
    if (b.contexts == 0) return true;
    const xdna_context::wkey key = { w->buffer, w->data, (int) w->type, w->ne[0], w->ne[1] };
    if (b.in.count(key)) return true;
    if (b.out.count(key)) return false;
    if (!b.started) {
        b.started = true;
        MEMORYSTATUSEX m = {};
        m.dwLength = sizeof(m);
        GlobalMemoryStatusEx(&m);
        b.free = (double) m.ullAvailPhys;
        b.kept_back = std::max(4 * GB, (double) m.ullTotalPhys / 10);
        const char * s = xdna_env("GGML_XDNA_MAX_COPY_GB");
        b.overridden = s != nullptr;
        b.limit = s ? atof(s) * GB : b.free - b.kept_back;
    }
    const xdna_budget::entry e = { xdna_layer_of(w), xdna_copy_bytes(w->ne[0], w->ne[1]) };
    if (!b.cut && b.taken + e.bytes <= b.limit) {
        b.in[key] = e;
        b.taken += e.bytes;
        return true;
    }
    if (!b.cut) {
        b.cut = true;
        // The layer's weights taken so far go back to the GPU with it, so no
        // layer is split between the two. Once a prompt has run, their copies
        // may exist, and they stay.
        if (e.layer >= 0 && !b.running)
            for (auto it = b.in.begin(); it != b.in.end();) {
                if (it->second.layer != e.layer) { ++it; continue; }
                b.taken -= it->second.bytes;
                b.out.insert(*it);
                it = b.in.erase(it);
            }
        std::unordered_set<int> layers;
        for (const auto & kv : b.in)
            if (kv.second.layer >= 0) layers.insert(kv.second.layer);
        char split[64] = "no layer fits, so all run on the GPU";
        if (!layers.empty()) snprintf(split, sizeof(split), "the first %zu layers on the NPU, the rest on the GPU", layers.size());
        GGML_LOG_WARN("xdna: NPU weight copies limited to %.1f GB (%s): %s\n", std::max(0.0, b.limit) / GB,
                      b.source().c_str(), split);
    }
    b.out[key] = e;
    return false;
}

// When nothing was left to the GPU, one line on what the copies take, as a
// context first runs a piece. (A limit reached says so where it's reached.)
static void xdna_budget_report() {
    xdna_budget & b = g_budget;
    std::lock_guard<std::mutex> lk(b.mu);
    if (b.running) return;
    b.running = true;
    if (!b.started || b.cut) return;
    GGML_LOG_INFO("xdna: NPU weight copies: %.1f GB for %zu weights (limit %.1f GB: %s)\n", b.taken / GB, b.in.size(),
                  std::max(0.0, b.limit) / GB, b.source().c_str());
}

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    return GGML_XDNA_DEVICE_NAME;
    GGML_UNUSED(backend);
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    delete (xdna_context *) backend->context;
    delete backend;
    std::lock_guard<std::mutex> lk(g_budget.mu);
    if (--g_budget.contexts == 0) g_budget.reset();
}

static bool xdna_mul_mat(xdna_context & ctx, ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0], * src1 = node->src[1];
    void * x = ctx.pinned(ggml_nbytes(src1));
    if (!x) {
        GGML_LOG_ERROR("%s: cannot allocate %zu bytes of pinned memory\n", __func__, ggml_nbytes(src1));
        return false;
    }
    auto t0 = std::chrono::steady_clock::now();
    auto since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    ctx.read(src1, x);
    ctx.read_ms += since(t0);
    ctx.out.resize(ggml_nbytes(node));
    ctx.matmuls++;

#ifdef XDNA_HAVE_NPU
    // On the NPU, for weights llama.cpp loaded (a weight buffer) in a built
    // shape: its 8-bit copy is made the first time it's used.
    xdna_npu * npu = ctx.get_npu();
    if (npu && src0->buffer && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        xdna_npu_has_shape(src0->ne[0], src0->ne[1])) {
        const xdna_npu::wkey key = { src0->buffer, src0->data, (int) src0->type, src0->ne[0], src0->ne[1] };
        std::string err;
        bool ok = npu->has_weight(key);
        if (!ok) {
            ctx.scratch_weight.resize(ggml_nbytes(src0));
            ctx.read(src0, ctx.scratch_weight.data());
            ok = npu->add_weight(key, ctx.scratch_weight.data(), err);
        }
        if (ok) ok = npu->mul_mat(key, (const float *) x, src1->ne[1], (float *) ctx.out.data(), err);
        if (ok) {
            t0 = std::chrono::steady_clock::now();
            ggml_backend_tensor_set(node, ctx.out.data(), 0, ggml_nbytes(node));
            ctx.write_ms += since(t0);
            return true;
        }
        xdna_npu_failed(std::string(node->name) + ": " + err);  // and on to the reference below
    }
#endif
    const std::vector<uint8_t> & w = ctx.weight(src0);

    // host-memory stand-ins for the three tensors, for the reference matmul
    ggml_tensor s0 = *src0, s1 = *src1, d = *node;
    s0.data = (void *) w.data();
    s1.data = x;
    d.data = ctx.out.data();
    xdna_ref_mul_mat(&s0, &s1, &d, xdna_n_threads());

    ggml_backend_tensor_set(node, ctx.out.data(), 0, ggml_nbytes(node));
    return true;
}

static bool xdna_trace();
static void xdna_trace_piece(const ggml_cgraph * cgraph);
static bool xdna_blocks();
static bool xdna_read_outside(const ggml_tensor * t, const std::unordered_set<const ggml_tensor *> & in_piece);

// How many streams block claiming splits a piece's rows into.
static int xdna_streams() {
    static int v = (int) env_int("GGML_XDNA_STREAMS", 2);
    return v;
}

static ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    xdna_context & ctx = *(xdna_context *) backend->context;
    xdna_budget_report();
    if (xdna_trace()) xdna_trace_piece(cgraph);
    const auto t_start = std::chrono::steady_clock::now();
    struct timer {
        xdna_context & c;
        std::chrono::steady_clock::time_point t;
        ~timer() {
            c.compute_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
            c.pieces++;
            if (!xdna_trace() || c.matmuls < 196) return;
#ifdef XDNA_HAVE_NPU
            const xdna_npu::times n = c.npu ? c.npu->take_times() : xdna_npu::times{};
#else
            const struct { double encode_ms = 0, npu_ms = 0, decode_ms = 0, pack_ms = 0; int calls = 0; } n;
#endif
            fprintf(stderr,
                    "xdna times, %lld matmuls in %lld pieces: all %.1f ms | read inputs %.1f, encode %.1f, NPU %.1f "
                    "(%d calls), decode %.1f, write results %.1f, weight copies %.1f ms\n",
                    (long long) c.matmuls, (long long) c.pieces, c.compute_ms, c.read_ms, n.encode_ms, n.npu_ms, n.calls,
                    n.decode_ms, c.write_ms, n.pack_ms);
#ifdef XDNA_HAVE_NPU
            // block claiming reads NPU outputs and encodes inputs inside the
            // passes, so there "encode" is only padding and copies, and
            // "decode" is 0
            fprintf(stderr, "xdna host passes: %.1f ms in %lld\n", xdna_pass_ms, (long long) xdna_passes);
            xdna_pass_ms = 0;
            xdna_passes = 0;
#endif
            c.read_ms = c.write_ms = c.compute_ms = 0;
            c.matmuls = c.pieces = 0;
        }
    } tm{ ctx, t_start };

#ifdef XDNA_HAVE_NPU
    // Block claiming runs every matmul on the NPU. The policy only hands us
    // matmuls it can run, but a caller computing a graph directly (a test)
    // can pass others; a graph of only such matmuls goes op by op below,
    // which falls back to the CPU. A graph with small ops in it is a claimed
    // block: it runs here, on the host if the NPU has failed.
    bool npu_takes_all = true, small_ops = false;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op != GGML_OP_MUL_MAT) {
            small_ops |= n->op != GGML_OP_NONE && n->op != GGML_OP_RESHAPE;
            continue;
        }
        const ggml_tensor * w = n->src[0];
        npu_takes_all &= w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                         xdna_npu_has_shape(w->ne[0], w->ne[1]);
    }
    if (xdna_blocks() && (npu_takes_all || small_ops)) {
        xdna_npu * npu = npu_takes_all ? ctx.get_npu() : nullptr;
        std::unordered_set<const ggml_tensor *> in_piece;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            in_piece.insert(cgraph->nodes[i]);
            ctx.matmuls += cgraph->nodes[i]->op == GGML_OP_MUL_MAT;
        }
        auto since = [](std::chrono::steady_clock::time_point t) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        };
        xdna_io io;
        io.read = [&](const ggml_tensor * t, void * dst) {
            const auto t0 = std::chrono::steady_clock::now();
            ctx.read(t, dst);
            ctx.read_ms += since(t0);
        };
        io.write = [&](ggml_tensor * t, const void * src) {
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
            ctx.write_ms += since(t0);
        };
        io.param = [&](const ggml_tensor * t) { return ctx.param(t); };
        io.alloc = [&](size_t n) { return ctx.pool_alloc(n); };
        io.needed_outside = [&](const ggml_tensor * t, const ggml_cgraph *) { return xdna_read_outside(t, in_piece); };
        io.ensure_weight = [&](const ggml_tensor * w, std::string & err) {
            if (!w->buffer || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                err = std::string(w->name) + " is not in a weight buffer";
                return false;
            }
            const xdna_npu::wkey key = { w->buffer, w->data, (int) w->type, w->ne[0], w->ne[1] };
            if (npu->has_weight(key)) return true;
            ctx.scratch_weight.resize(ggml_nbytes(w));
            ctx.read(w, ctx.scratch_weight.data());
            return npu->add_weight(key, ctx.scratch_weight.data(), err);
        };
        io.ensure_fused = [&](const ggml_tensor * gw, const ggml_tensor * uw, std::string & err) {
            for (const ggml_tensor * w : { gw, uw })
                if (!w->buffer || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    err = std::string(w->name) + " is not in a weight buffer";
                    return false;
                }
            const xdna_npu::wkey gate = { gw->buffer, gw->data, (int) gw->type, gw->ne[0], gw->ne[1] };
            if (npu->has_weight(xdna_npu::fused_key(gate))) return true;
            std::vector<uint8_t> g(ggml_nbytes(gw)), u(ggml_nbytes(uw));
            ctx.read(gw, g.data());
            ctx.read(uw, u.data());
            return npu->add_fused(gate, g.data(), u.data(), err);
        };
        io.weight_bytes = [&](const ggml_tensor * w) -> const void * { return ctx.weight(w).data(); };
        std::string err;
        bool ok = false;
        if (npu) {
            ok = xdna_exec_piece(cgraph, npu, npu->pool(), io, xdna_streams(), err);
            ctx.pool_release();
            if (!ok) xdna_npu_failed(err);
        }
        if (!ok) {
            // a failed piece wrote nothing back; run it again on the host
            ok = xdna_exec_piece(cgraph, nullptr, ctx.host_threads(), io, 1, err);
            ctx.pool_release();
        }
        if (!ok) {
            GGML_LOG_ERROR("%s: %s\n", __func__, err.c_str());
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }
#endif

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            case GGML_OP_MUL_MAT:
                if (!xdna_mul_mat(ctx, node)) return GGML_STATUS_FAILED;
                break;
            default:
                GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_xdna_i = {
    /* .get_name             = */ ggml_backend_xdna_get_name,
    /* .free                 = */ ggml_backend_xdna_free,
    /* .set_tensor_async     = */ NULL,
    /* .get_tensor_async     = */ NULL,
    /* .set_tensor_2d_async  = */ NULL,
    /* .get_tensor_2d_async  = */ NULL,
    /* .cpy_tensor_async     = */ NULL,
    /* .synchronize          = */ NULL,
    /* .graph_plan_create    = */ NULL,
    /* .graph_plan_free      = */ NULL,
    /* .graph_plan_update    = */ NULL,
    /* .graph_plan_compute   = */ NULL,
    /* .graph_compute        = */ ggml_backend_xdna_graph_compute,
    /* .event_record         = */ NULL,
    /* .event_wait           = */ NULL,
    /* .graph_optimize       = */ NULL,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = { 0x9d, 0x3a, 0x1c, 0x74, 0x2e, 0x58, 0x41, 0xb6,
                              0x8f, 0x0d, 0xc5, 0x93, 0x6a, 0x27, 0xe1, 0x40 };
    return &guid;
}

//
// device
//

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    return GGML_XDNA_DEVICE_NAME;
    GGML_UNUSED(dev);
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    return XDNA_DESCRIPTION;
    GGML_UNUSED(dev);
}

// No free memory, but some in total: llama.cpp then gives this device no
// layers (all weights go to Vulkan). Zero for both would make it fall back to
// counting system memory, and hand us layers.
static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 1;
    GGML_UNUSED(dev);
}

// An integrated GPU, not an accelerator: llama.cpp ranks accelerators below
// every GPU, where the scheduler would never hand us anything.
static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_IGPU;
    GGML_UNUSED(dev);
}

static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna_device_get_name(dev);
    props->description = ggml_backend_xdna_device_get_description(dev);
    props->type        = ggml_backend_xdna_device_get_type(dev);
    props->device_id   = NULL;
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);

    // no async and no events: llama.cpp then keeps pipeline parallelism off,
    // and hands over between us and Vulkan with plain synchronizes
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_t ggml_backend_xdna_device_init(ggml_backend_dev_t dev, const char * params) {
    const xdna_vk & vk = xdna_vulkan();
    if (!vk.dev) return nullptr;
    auto * ctx = new xdna_context;
    ctx->vk = ggml_backend_dev_init(vk.dev, nullptr);
    if (!ctx->vk) {
        GGML_LOG_ERROR("%s: cannot start a backend on %s\n", __func__, ggml_backend_dev_name(vk.dev));
        delete ctx;
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(g_budget.mu);
        g_budget.contexts++;
    }
    return new ggml_backend{
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ ggml_backend_xdna_i,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
    GGML_UNUSED(params);
}

// The weight side of a matmul we can take: a plain 2D weight tensor (not a
// view, not computed), of a type ggml can unpack. Keys and values from the
// cache are views, so attention's own matmuls never match.
static bool xdna_is_weight(const ggml_tensor * w) {
    if (w->op != GGML_OP_NONE || w->view_src != nullptr) return false;
    if (w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(w)) return false;
    if (w->type == GGML_TYPE_F32) return true;
    const ggml_type_traits * tr = ggml_get_type_traits(w->type);
    return tr && tr->to_float != nullptr;
}

// Who reads what, as the scheduler shows us. It asks supports_op about every
// op it places: the ones we take, and every one it assigns to Vulkan (we rank
// above Vulkan and share its buffer type, so its upgrade pass checks us
// first). Each question carries the op's inputs, so this records every reader
// of every tensor. Block claiming writes a result back to Vulkan only when a
// reader outside the piece exists.
// Readers from earlier graphs whose tensors ggml reused can linger; that only
// ever adds a write, never skips one.
static bool xdna_trace() {
    static bool v = env_int("GGML_XDNA_TRACE", 0) != 0;
    return v;
}
static std::mutex g_read_mu;
static std::unordered_map<const ggml_tensor *, std::unordered_set<const ggml_tensor *>> g_readers;

static void xdna_note_readers(const ggml_tensor * op) {
    std::lock_guard<std::mutex> lk(g_read_mu);
    for (int j = 0; j < GGML_MAX_SRC; j++)
        if (const ggml_tensor * s = op->src[j]) {
            g_readers[s].insert(op);
            for (const ggml_tensor * v = s->view_src; v; v = v->view_src) g_readers[v].insert(op);
        }
}

// Whether something outside `in_piece` reads `t`, or might: a graph output, a
// tensor never seen, or one with a reader elsewhere.
static bool xdna_read_outside(const ggml_tensor * t, const std::unordered_set<const ggml_tensor *> & in_piece) {
    static const bool all = env_int("GGML_XDNA_WRITE_ALL", 0) != 0;  // diagnostic: write every result back
    if (all || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) return true;
    std::lock_guard<std::mutex> lk(g_read_mu);
    auto it = g_readers.find(t);
    if (it == g_readers.end()) return true;
    for (const ggml_tensor * r : it->second)
        if (!in_piece.count(r)) return true;
    return false;
}

static void xdna_trace_piece(const ggml_cgraph * cgraph) {
    static int64_t pieces = 0, results = 0, unseen = 0;
    pieces++;
    std::unordered_set<const ggml_tensor *> in_piece;
    for (int i = 0; i < cgraph->n_nodes; i++) in_piece.insert(cgraph->nodes[i]);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op != GGML_OP_MUL_MAT) continue;
        results++;
        std::lock_guard<std::mutex> lk(g_read_mu);
        if (!g_readers.count(n)) {
            unseen++;
            fprintf(stderr, "xdna trace: %s (%s) has no recorded reader\n", n->name, ggml_op_name(n->op));
        }
    }
    if (pieces % 100 == 0)
        fprintf(stderr, "xdna trace: %lld pieces, %lld matmul results, %lld with no recorded reader\n", (long long) pieces,
                (long long) results, (long long) unseen);
}

// Block claiming: with the NPU configured, also take the small
// ops next to our matmuls. GGML_XDNA_BLOCKS=0 claims matmuls only.
static bool xdna_blocks() {
    static bool v = env_int("GGML_XDNA_BLOCKS", 1) != 0;
#ifdef XDNA_HAVE_NPU
    return v && xdna_run_mode() == xdna_mode::npu;
#else
    return false;
#endif
}

// Rows are prompt tokens, along a tensor's last dimension in use.
static int64_t xdna_tokens(const ggml_tensor * t) {
    return t->ne[3] > 1 ? t->ne[3] : t->ne[2] > 1 ? t->ne[2] : t->ne[1];
}

static bool xdna_supports_op_policy(const ggml_tensor * op, int depth = 0);

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    if (xdna_blocks() || xdna_trace()) xdna_note_readers(op);
    return xdna_supports_op_policy(op);
    GGML_UNUSED(dev);
}

static bool xdna_claim_mul_mat(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0], * src1 = op->src[1];
    if (xdna_op_batch_size(op) < xdna_min_batch()) return false;
    if (!xdna_is_weight(src0)) return false;
    if (op->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(src1) || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src0->ne[0] != src1->ne[0]) return false;
#ifdef XDNA_HAVE_NPU
    // on the NPU, only sizes the kernel takes
    if (xdna_run_mode() == xdna_mode::npu && !xdna_npu_has_shape(src0->ne[0], src0->ne[1])) return false;
#endif
    const int64_t mflop = 2 * src0->ne[0] * src0->ne[1] * xdna_op_batch_size(op) / 1000000;
    return mflop >= xdna_min_mflop() && xdna_budget_takes(src0);
}

// The whole policy. Anything accepted here is taken from Vulkan, so it has to
// be strict. It must also cope with llama.cpp's load-time probe, where the
// weight has a placeholder buffer and no data: nothing here reads data.
//
// A small op is claimed only when the executor implements it, it covers a big
// batch, and one of its inputs comes from an op we'd claim: that grows pieces
// out from our matmuls (norm -> q/k/v; o -> add -> norm -> gate/up -> SiLU ->
// down -> add) without pulling in unrelated work, which would split Vulkan's
// graph into pieces for nothing.
static bool xdna_supports_op_policy(const ggml_tensor * op, int depth) {
    if (!xdna_vulkan().dev || g_npu_broken) return false;
    if (op->op == GGML_OP_MUL_MAT) return xdna_claim_mul_mat(op);
#ifdef XDNA_HAVE_NPU
    if (!xdna_blocks() || depth > 8) return false;
    if (op->op != GGML_OP_RESHAPE && xdna_tokens(op) < xdna_min_batch()) return false;
    if (!xdna_exec_supports(op)) return false;
    {
        // GGML_XDNA_BLOCK_OPS: a comma list of the small ops block claiming
        // may take (ggml's op names, e.g. "ADD,RMS_NORM"), for tracking down
        // a wrong result op by op. Unset: all of them.
        static const std::string allow = [] {
            const char * s = xdna_env("GGML_XDNA_BLOCK_OPS");
            return s ? "," + std::string(s) + "," : std::string();
        }();
        if (!allow.empty() && allow.find("," + std::string(ggml_op_name(op->op)) + ",") == std::string::npos) return false;
    }
    if (op->op == GGML_OP_MUL) {
        const ggml_tensor * w = op->src[1];  // a scale row: a plain leaf, kept once
        if (w->op != GGML_OP_NONE || w->view_src) return false;
    }
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        const ggml_tensor * s = op->src[j];
        if (s && s->op != GGML_OP_NONE && xdna_supports_op_policy(s, depth + 1)) return true;
    }
#endif
    return false;
    GGML_UNUSED(depth);
}

// Vulkan's buffers, and host memory too (the CPU's graph inputs: the token
// embeddings, positions). Reading host tensors directly also keeps our pieces
// free of scheduler-copied inputs, and that matters for correctness: before a
// piece with no inputs the scheduler waits for the previous backend (Vulkan),
// but before a piece with inputs it waits only for the backend each input
// came from. A piece whose one copied input came from the CPU could then
// start reading Vulkan results before Vulkan had finished them.
static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    const xdna_vk & vk = xdna_vulkan();
    return (vk.buft != nullptr && buft == vk.buft) || ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

// Vulkan's own buffer type: sharing it is what lets the scheduler hand ops
// between us and Vulkan with no copies.
static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    const xdna_vk & vk = xdna_vulkan();
    if (!vk.buft) GGML_ABORT(GGML_XDNA_DEVICE_NAME " needs the Vulkan backend loaded");
    return vk.buft;
    GGML_UNUSED(dev);
}

static const ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    // deliberately no offload_op: that path copies the weights per op
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// reg
//

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_XDNA_NAME;
    GGML_UNUSED(reg);
}

// No device when the NPU can't run the backend (and host-only mode is off):
// llama.cpp then never sees XDNA0, and runs as if the backend weren't there.
static size_t ggml_backend_xdna_reg_device_count(ggml_backend_reg_t reg) {
    return xdna_run_mode() == xdna_mode::off ? 0 : 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xdna_reg_device_get(ggml_backend_reg_t reg, size_t index);

static const ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_device_get,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ NULL,
    };
    return &reg;
}

static ggml_backend_dev_t ggml_backend_xdna_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_xdna_device_i,
        /* .reg     = */ ggml_backend_xdna_reg(),
        /* .context = */ NULL,
    };
    return &dev;
    GGML_UNUSED(reg);
}

static int ggml_backend_xdna_score(void) {
    return 1;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_xdna_score)
