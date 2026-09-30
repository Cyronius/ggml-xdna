#include "xdna-npu.h"

#include "xdna-env.h"

#include "bfp16_insts.h"
#include "bfp16_pack.h"
#include "thread_pool.h"
#include "xrt_shim.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <tuple>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

constexpr int64_t SMALL = 512, BIG = 1024;  // rows per call
const bfp16_tiling TILE;                      // 128 x 64 x 64, 8 columns: what the builds use
const uint8_t zero_block[BFP16_BLOCK_BYTES] = {};

// The kernel takes this size at both row counts.
bool fits(int64_t K, int64_t N) {
    return !xdna_xclbin().empty() && bfp16_insts_fits(SMALL, K, N) && bfp16_insts_fits(BIG, K, N);
}

// The kernel's N comes in multiples of 512 (64 per core column). A weight of
// another width gets zero rows up to the next multiple, and only its real
// columns are read back. A fused gate/up pads F to a multiple of 256, so 2F
// stays a multiple of 512.
int64_t pad_n(int64_t N) { return (N + 511) / 512 * 512; }
int64_t pad_f(int64_t F) { return (F + 255) / 256 * 256; }

bool out16_enabled() {
    static const bool v = [] {
        const char * s = xdna_env("GGML_XDNA_OUT16");
        return !s || atoi(s) != 0;
    }();
    return v;
}

// One row of an NPU output, in its mode, into `row` (N values, or N/2 in mode
// 2), of which the first `n_out` are kept.
void read_row(const float * c, int64_t M, int64_t N, int mode, int64_t r, int64_t n_out, float * row) {
    const int64_t n_all = mode == 2 ? N / 2 : N;
    float * dst = row;
    thread_local std::vector<float> padded;
    if (n_out < n_all) {
        padded.resize((size_t) n_all);
        dst = padded.data();
    }
    if (mode) bfp16_c16_row(c, M, N, TILE, mode, r, dst);
    else bfp16_c_row(c, M, N, TILE, r, N, dst);
    if (dst != row) memcpy(row, dst, (size_t) n_out * sizeof(float));
}

// GGML_XDNA_FAIL_AFTER=n: every NPU submission after the n-th fails, to test
// the backend's fallback to the CPU.
bool injected_failure(std::string & err) {
    static const long long after = [] {
        const char * s = xdna_env("GGML_XDNA_FAIL_AFTER");
        return s ? atoll(s) : -1LL;
    }();
    static std::atomic<long long> n{ 0 };
    if (after < 0 || ++n <= after) return false;
    err = "an injected failure (GGML_XDNA_FAIL_AFTER)";
    return true;
}

} // namespace

// The directory this DLL was loaded from.
static std::filesystem::path module_dir() {
    HMODULE h = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR) &module_dir, &h))
        return {};
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(h, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) return {};
    return std::filesystem::path(std::wstring(buf, n)).parent_path();
}

const std::string & xdna_xclbin() {
    static const std::string path = []() -> std::string {
        namespace fs = std::filesystem;
        std::error_code ec;
        const char * s = xdna_env("GGML_XDNA_KERNELS");
        if (!s || !*s) {
            const fs::path p = module_dir() / "bfp16_gemm.xclbin";
            return fs::is_regular_file(p, ec) ? p.u8string() : std::string();
        }
        const fs::path p = fs::u8path(s);
        if (fs::is_regular_file(p, ec)) return p.u8string();
        if (fs::is_regular_file(p / "final.xclbin", ec)) return (p / "final.xclbin").u8string();
        // a directory of per-size builds: every one holds the same program
        for (const auto & e : fs::directory_iterator(p, ec))
            if (fs::is_regular_file(e.path() / "final.xclbin", ec)) return (e.path() / "final.xclbin").u8string();
        return {};
    }();
    return path;
}

// NPUs the backend was tested on, as XRT names them: "NPU Strix" is Strix
// Point (Ryzen AI 300, e.g. the HX 370).
static const char * const TESTED_NPUS[] = { "NPU Strix" };

bool xdna_npu_usable(std::string & why) {
    static std::string reason;
    static const bool ok = [] {
        if (xdna_xclbin().empty()) {
            const char * s = xdna_env("GGML_XDNA_KERNELS");
            reason = s && *s ? std::string("GGML_XDNA_KERNELS=") + s + " names no xclbin"
                             : "no bfp16_gemm.xclbin next to ggml-xdna.dll";
            return false;
        }
        xrtsh_dev dev = xrtsh_device_open(0);
        if (!dev) {
            reason = std::string("no NPU found (") + xrtsh_last_error() + ")";
            return false;
        }
        char name[256] = {};
        xrtsh_device_name(dev, name, sizeof(name));
        const char * any = xdna_env("GGML_XDNA_ANY_NPU");
        bool tested = any && atoi(any) != 0;
        for (const char * t : TESTED_NPUS) tested |= strcmp(name, t) == 0;
        if (!tested) {
            reason = std::string("the NPU \"") + name + "\" hasn't been tested with this backend (GGML_XDNA_ANY_NPU=1 tries it)";
            xrtsh_device_free(dev);
            return false;
        }
        xrtsh_ctx ctx = xrtsh_hwctx_create(dev, xdna_xclbin().c_str());
        if (!ctx) {
            reason = "the NPU won't load " + xdna_xclbin() + " (" + xrtsh_last_error() + ")";
            xrtsh_device_free(dev);
            return false;
        }
        xrtsh_hwctx_free(ctx);
        xrtsh_device_free(dev);
        reason = std::string("NPU \"") + name + "\", " + xdna_xclbin();
        return true;
    }();
    why = reason;
    return ok;
}

bool xdna_npu_has_shape(int64_t K, int64_t N) { return fits(K, pad_n(N)); }
bool xdna_npu::has_fused_shape(int64_t K, int64_t F) { return out16_enabled() && fits(K, 2 * pad_f(F)); }

bool xdna_npu::wkey::operator<(const wkey & o) const {
    if (buffer != o.buffer) return buffer < o.buffer;
    if (data != o.data) return data < o.data;
    if (type != o.type) return type < o.type;
    if (K != o.K) return K < o.K;
    return N < o.N;
}

xdna_npu::xdna_npu() = default;
xdna_npu::~xdna_npu() {
    flights_.clear();  // batches go before the device they run on
    batch_.reset();
    npu_.reset();
}

bool xdna_npu::init(int n_threads, std::string & err) {
    if (xdna_xclbin().empty()) { err = "GGML_XDNA_KERNELS names no xclbin"; return false; }
    npu_ = std::make_unique<npu_bfp16>();
    pool_ = std::make_unique<thread_pool>(n_threads);
    batch_ = std::make_unique<npu_bfp16::batch>(*npu_);
    return true;
}

// A size's instruction stream, made once. The first also opens the device
// with the xclbin.
int xdna_npu::shape(int64_t M, int64_t K, int64_t N, int mode, std::string & err) {
    const auto key = std::make_tuple(M, K, N, mode);
    auto it = shapes_.find(key);
    if (it != shapes_.end()) return it->second;
    if (!opened_) {
        if (!npu_->open(xdna_xclbin(), err)) return -1;
        opened_ = true;
    }
    const std::vector<uint32_t> words = bfp16_insts(M, K, N, mode);
    if (words.empty()) {
        err = "the kernel doesn't take " + std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);
        return -1;
    }
    const int h = npu_->add_shape(words, M, K, N, err);
    if (h >= 0) shapes_[key] = h;
    return h;
}

void xdna_npu::unpack(const void * raw, ggml_type type, int64_t N, int64_t K, float * dst) {
    const size_t row_bytes = ggml_row_size(type, K);
    pool_->parallel_for(N, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; r++) {
            const uint8_t * src = (const uint8_t *) raw + r * row_bytes;
            if (type == GGML_TYPE_F32) memcpy(dst + r * K, src, (size_t) K * sizeof(float));
            else ggml_get_type_traits(type)->to_float(src, dst + r * K, K);
        }
    });
}

// f: N x K float32 rows (N padded), encoded and laid out for the kernel;
// n_out of each output row's values are real.
bool xdna_npu::add_packed(const wkey & key, std::vector<float> & f, int64_t N, int64_t n_out, int mode, std::string & err) {
    weight w;
    w.K = key.K;
    w.N = N;
    w.mode = mode;
    w.N_out = n_out;
    w.shape_small = shape(SMALL, key.K, N, mode, err);
    w.shape_big = shape(BIG, key.K, N, mode, err);
    if (w.shape_small < 0 || w.shape_big < 0) return false;
    std::vector<uint8_t> packed;
    bfp16_pack_b(f.data(), N, key.K, TILE, packed, pool_->size());
    w.handle = npu_->add_weights(packed, err);
    if (w.handle < 0) return false;
    weights_[key] = w;
    return true;
}

bool xdna_npu::add_weight(const wkey & key, const void * raw, std::string & err) {
    const clk::time_point t0 = clk::now();
    const int64_t N = pad_n(key.N);
    std::vector<float> f((size_t) (N * key.K), 0.0f);
    unpack(raw, (ggml_type) key.type, key.N, key.K, f.data());
    const int mode = out16_enabled() ? 1 : 0;
    const bool ok = add_packed(key, f, N, key.N, mode, err);
    times_.pack_ms += ms_since(t0);
    return ok;
}

bool xdna_npu::add_fused(const wkey & gate, const void * gate_raw, const void * up_raw, std::string & err) {
    const clk::time_point t0 = clk::now();
    const int64_t F = gate.N, K = gate.K, Fp = pad_f(F);
    std::vector<float> g((size_t) (Fp * K), 0.0f), u((size_t) (Fp * K), 0.0f), f((size_t) (2 * Fp * K));
    unpack(gate_raw, (ggml_type) gate.type, F, K, g.data());
    unpack(up_raw, (ggml_type) gate.type, F, K, u.data());
    bfp16_interleave_gate_up(g.data(), u.data(), Fp, K, f.data());
    const bool ok = add_packed(fused_key(gate), f, 2 * Fp, F, 2, err);
    times_.pack_ms += ms_since(t0);
    return ok;
}

bool xdna_npu::mul_mat(const wkey & key, const float * x, int64_t T, float * y, std::string & err) {
    const weight & w = weights_.at(key);
    const int64_t K = w.K, N = w.N;
    const int n = (int) ((T + BIG - 1) / BIG);
    auto rows_of = [&](int j) { return std::min(BIG, T - j * BIG); };
    auto m_of = [&](int j) { return rows_of(j) > SMALL ? BIG : SMALL; };

    clk::time_point t0 = clk::now();
    std::vector<int> shapes;
    for (int j = 0; j < n; j++) shapes.push_back(m_of(j) == BIG ? w.shape_big : w.shape_small);
    if (!batch_->reserve(n, (size_t) (BIG * K / 8) * BFP16_BLOCK_BYTES, 0, (size_t) (BIG * N) * sizeof(float), err)) return false;
    if (!batch_->prepare(shapes, err, { w.handle })) return false;
    // every call's rows, encoded straight into its buffer; rows past the
    // prompt as zeros
    pool_->parallel_for((int64_t) n * BIG, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; i++) {
            const int j = (int) (i / BIG);
            const int64_t r = i % BIG;
            if (r >= m_of(j)) continue;
            uint8_t * A = batch_->a(j);
            const int64_t t = j * BIG + r;
            if (t < T) {
                const float * row = x + t * K;
                bfp16_a_row(r, K, TILE, [&](int64_t cb, size_t off) { bfp16_encode_block(row + cb * 8, A + off); });
            } else {
                bfp16_a_row(r, K, TILE, [&](int64_t, size_t off) { memcpy(A + off, zero_block, sizeof(zero_block)); });
            }
        }
    });
    times_.encode_ms += ms_since(t0);

    t0 = clk::now();
    if (injected_failure(err) || !batch_->run(err)) return false;
    times_.npu_ms += ms_since(t0);
    times_.calls += n;

    t0 = clk::now();
    pool_->parallel_for(T, [&](int64_t t0r, int64_t t1r) {
        for (int64_t t = t0r; t < t1r; t++) {
            const int j = (int) (t / BIG);
            read_row(batch_->c(j), m_of(j), N, w.mode, t % BIG, w.N_out, y + t * w.N_out);
        }
    });
    times_.decode_ms += ms_since(t0);
    return true;
}

xdna_npu::flight & xdna_npu::flight_for(int stream) {
    if ((int) flights_.size() <= stream) flights_.resize(stream + 1);
    flight & f = flights_[stream];
    if (!f.batch) f.batch = std::make_unique<npu_bfp16::batch>(*npu_);
    return f;
}

// Call i's output buffer is sized for the widest i-th weight of any group.
// A batch grows each call's buffers to what it's asked for, so asking for
// calls 0..i at call i's width gives call j the widest of calls j and up.
bool xdna_npu::reserve(int stream, const std::vector<std::vector<wkey>> & groups, std::string & err) {
    flight & f = flight_for(stream);
    size_t a_bytes = 0;
    std::vector<size_t> c_bytes;
    for (const auto & keys : groups)
        for (size_t i = 0; i < keys.size(); i++) {
            const weight & w = weights_.at(keys[i]);
            a_bytes = std::max(a_bytes, (size_t) (BIG * w.K / 8) * BFP16_BLOCK_BYTES);
            if (c_bytes.size() <= i) c_bytes.resize(i + 1, 0);
            c_bytes[i] = std::max(c_bytes[i], (size_t) (BIG * w.N) * sizeof(float));
        }
    f.readable = false;  // growing a buffer loses what it held
    if (f.c_cap.size() < c_bytes.size()) f.c_cap.resize(c_bytes.size(), 0);
    for (size_t i = 0; i < c_bytes.size(); i++) {
        if (!f.batch->reserve((int) i + 1, a_bytes, 0, c_bytes[i], err)) return false;
        for (size_t j = 0; j <= i; j++) f.c_cap[j] = std::max(f.c_cap[j], c_bytes[i]);
    }
    return true;
}

bool xdna_npu::begin(int stream, const std::vector<wkey> & keys, int64_t rows, std::string & err) {
    if (rows > BIG || keys.empty()) { err = "bad group"; return false; }
    flight & f = flight_for(stream);
    const int64_t M = rows > SMALL ? BIG : SMALL;
    const int64_t K = weights_.at(keys[0]).K;
    std::vector<const weight *> ws;
    std::vector<int> shapes, handles;
    for (size_t i = 0; i < keys.size(); i++) {
        const weight & w = weights_.at(keys[i]);
        if (w.K != K) { err = "group weights differ in K"; return false; }
        // preparing a call whose buffer is too small moves it, and the
        // previous group's output with it
        if (f.readable && i < f.w_out.size() && (i >= f.c_cap.size() || (size_t) (M * w.N) * sizeof(float) > f.c_cap[i])) {
            err = "a group's output buffer was not reserved";
            return false;
        }
        ws.push_back(&w);
        shapes.push_back(M == BIG ? w.shape_big : w.shape_small);
        handles.push_back(w.handle);
    }
    const clk::time_point t0 = clk::now();
    if (!f.batch->prepare(shapes, err, handles)) return false;
    f.w = std::move(ws);
    f.rows = rows;
    f.M = M;
    // rows past the prompt: zeros (the buffer held other data)
    uint8_t * A = f.batch->a(0);
    pool_->parallel_for(M - rows, [&](int64_t i0, int64_t i1) {
        for (int64_t r = rows + i0; r < rows + i1; r++)
            bfp16_a_row(r, K, TILE, [&](int64_t, size_t off) { memcpy(A + off, zero_block, sizeof(zero_block)); });
    });
    times_.encode_ms += ms_since(t0);
    return true;
}

void xdna_npu::encode_row(int stream, int64_t r, const float * row) {
    const flight & f = flights_[stream];
    uint8_t * A = f.batch->a(0);
    bfp16_a_row(r, f.w[0]->K, TILE, [&](int64_t cb, size_t off) { bfp16_encode_block(row + cb * 8, A + off); });
}

bool xdna_npu::submit(int stream, std::string & err) {
    if (injected_failure(err)) return false;
    flight & f = flights_.at(stream);
    clk::time_point t0 = clk::now();
    // every call reads the same input
    const size_t a_bytes = (size_t) (f.M * f.w[0]->K / 8) * BFP16_BLOCK_BYTES;
    for (size_t i = 1; i < f.w.size(); i++) memcpy(f.batch->a((int) i), f.batch->a(0), a_bytes);
    times_.encode_ms += ms_since(t0);
    f.readable = false;
    t0 = clk::now();
    if (!f.batch->start(err)) return false;
    times_.npu_ms += ms_since(t0);  // syncing the operands and submitting
    times_.calls += (int) f.w.size();
    return true;
}

bool xdna_npu::wait(int stream, std::string & err) {
    flight & f = flights_.at(stream);
    const clk::time_point t0 = clk::now();
    if (!f.batch->wait(err)) return false;
    times_.npu_ms += ms_since(t0);  // what the host actually waited
    f.w_out = f.w;
    f.M_out = f.M;
    f.readable = true;
    return true;
}

void xdna_npu::decode_row(int stream, int k, int64_t r, float * row) const {
    const flight & f = flights_[stream];
    const weight & w = *f.w_out[k];
    read_row(f.batch->c(k), f.M_out, w.N, w.mode, r, w.N_out, row);
}

xdna_npu::times xdna_npu::take_times() {
    times t = times_;
    times_ = {};
    return t;
}
