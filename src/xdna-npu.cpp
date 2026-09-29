#include "xdna-npu.h"

#include "bfp16_pack.h"
#include "thread_pool.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

constexpr int64_t SMALL = 512, BIG = 1024;  // rows per call
const bfp16_tiling TILE;                      // 128 x 64 x 64, 8 columns: what the builds use
const uint8_t zero_block[BFP16_BLOCK_BYTES] = {};

// A build's directory: <M>x<K>x<N>_128x64x64_c8, with _m<mode> for the 16-bit
// output modes (kernels/bfp16_gemm/build.ps1 -OutMode).
std::string build_dir(int64_t M, int64_t K, int64_t N, int mode = 0) {
    return xdna_kernels_dir() + "/" + std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N) +
           "_128x64x64_c8" + (mode ? "_m" + std::to_string(mode) : "");
}

bool file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f) fclose(f);
    return f != nullptr;
}

// Built at both row counts, in this mode. Cached; thread-safe.
bool built(int64_t K, int64_t N, int mode) {
    if (xdna_kernels_dir().empty()) return false;
    static std::mutex mu;
    static std::map<std::tuple<int64_t, int64_t, int>, bool> known;
    std::lock_guard<std::mutex> lk(mu);
    auto key = std::make_tuple(K, N, mode);
    auto it = known.find(key);
    if (it != known.end()) return it->second;
    const bool ok =
        file_exists(build_dir(SMALL, K, N, mode) + "/insts.bin") && file_exists(build_dir(BIG, K, N, mode) + "/insts.bin");
    known[key] = ok;
    return ok;
}

bool out16_enabled() {
    static const bool v = [] {
        const char * s = getenv("GGML_XDNA_OUT16");
        return !s || atoi(s) != 0;
    }();
    return v;
}

// One row of an NPU output, in its mode, into `row` (N values, or N/2 in mode 2).
void read_row(const float * c, int64_t M, int64_t N, int mode, int64_t r, float * row) {
    if (mode) bfp16_c16_row(c, M, N, TILE, mode, r, row);
    else bfp16_c_row(c, M, N, TILE, r, N, row);
}

} // namespace

const std::string & xdna_kernels_dir() {
    static const std::string dir = [] {
        const char * s = getenv("GGML_XDNA_KERNELS");
        std::string d = s ? s : "";
        while (!d.empty() && (d.back() == '/' || d.back() == '\\')) d.pop_back();
        return d;
    }();
    return dir;
}

bool xdna_npu_has_shape(int64_t K, int64_t N) { return built(K, N, 0); }
bool xdna_npu::has_fused_shape(int64_t K, int64_t F) { return out16_enabled() && built(K, 2 * F, 2); }

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
    if (xdna_kernels_dir().empty()) { err = "GGML_XDNA_KERNELS is not set"; return false; }
    npu_ = std::make_unique<npu_bfp16>();
    pool_ = std::make_unique<thread_pool>(n_threads);
    batch_ = std::make_unique<npu_bfp16::batch>(*npu_);
    return true;
}

// A build's instruction stream, loaded once. The first build loaded also
// opens the device with its xclbin; every build shares that core program.
int xdna_npu::shape(int64_t M, int64_t K, int64_t N, int mode, std::string & err) {
    const std::string dir = build_dir(M, K, N, mode);
    auto it = shapes_.find(dir);
    if (it != shapes_.end()) return it->second;
    if (!opened_) {
        if (!npu_->open(dir + "/final.xclbin", err)) return -1;
        opened_ = true;
    }
    const int h = npu_->add_shape(dir + "/insts.bin", M, K, N, err);
    if (h >= 0) shapes_[dir] = h;
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

// f: N x K float32 rows, encoded and laid out for the kernel.
bool xdna_npu::add_packed(const wkey & key, std::vector<float> & f, int mode, std::string & err) {
    weight w;
    w.K = key.K;
    w.N = key.N;
    w.mode = mode;
    w.N_out = mode == 2 ? key.N / 2 : key.N;
    w.shape_small = shape(SMALL, key.K, key.N, mode, err);
    w.shape_big = shape(BIG, key.K, key.N, mode, err);
    if (w.shape_small < 0 || w.shape_big < 0) return false;
    std::vector<uint8_t> packed;
    bfp16_pack_b(f.data(), key.N, key.K, TILE, packed, pool_->size());
    w.handle = npu_->add_weights(packed, err);
    if (w.handle < 0) return false;
    weights_[key] = w;
    return true;
}

bool xdna_npu::add_weight(const wkey & key, const void * raw, std::string & err) {
    const clk::time_point t0 = clk::now();
    std::vector<float> f((size_t) (key.N * key.K));
    unpack(raw, (ggml_type) key.type, key.N, key.K, f.data());
    const int mode = out16_enabled() && built(key.K, key.N, 1) ? 1 : 0;
    const bool ok = add_packed(key, f, mode, err);
    times_.pack_ms += ms_since(t0);
    return ok;
}

bool xdna_npu::add_fused(const wkey & gate, const void * gate_raw, const void * up_raw, std::string & err) {
    const clk::time_point t0 = clk::now();
    const int64_t F = gate.N, K = gate.K;
    std::vector<float> g((size_t) (F * K)), u((size_t) (F * K)), f((size_t) (2 * F * K));
    unpack(gate_raw, (ggml_type) gate.type, F, K, g.data());
    unpack(up_raw, (ggml_type) gate.type, F, K, u.data());
    bfp16_interleave_gate_up(g.data(), u.data(), F, K, f.data());
    const bool ok = add_packed(fused_key(gate), f, 2, err);
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
    if (!batch_->run(err)) return false;
    times_.npu_ms += ms_since(t0);
    times_.calls += n;

    t0 = clk::now();
    pool_->parallel_for(T, [&](int64_t t0r, int64_t t1r) {
        for (int64_t t = t0r; t < t1r; t++) {
            const int j = (int) (t / BIG);
            read_row(batch_->c(j), m_of(j), N, w.mode, t % BIG, y + t * w.N_out);
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
    read_row(f.batch->c(k), f.M_out, w.N, w.mode, r, row);
}

xdna_npu::times xdna_npu::take_times() {
    times t = times_;
    times_ = {};
    return t;
}
