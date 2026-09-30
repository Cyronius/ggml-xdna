#include "npu_bfp16.h"

#include "xrt_shim.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <immintrin.h>

namespace {

std::string shim_error(const std::string & what) {
    return what + ": " + xrtsh_last_error();
}

bool read_file(const std::string & path, std::vector<uint8_t> & out, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

} // namespace

npu_bfp16::~npu_bfp16() { close(); }

void npu_bfp16::close() {
    if (run_) xrtsh_run_free(run_);
    for (shape & s : shapes_) if (s.ibo) xrtsh_bo_free(s.ibo);
    for (wbuf & w : weights_) if (w.bo) xrtsh_bo_free(w.bo);
    for (void * bo : abos_) if (bo) xrtsh_bo_free(bo);
    if (cbo_)  xrtsh_bo_free(cbo_);
    if (kern_) xrtsh_kernel_free(kern_);
    if (ctx_)  xrtsh_hwctx_free(ctx_);
    if (dev_)  xrtsh_device_free(dev_);
    shapes_.clear();
    weights_.clear();
    abos_.clear();
    a_caps_.clear();
    run_ = cbo_ = kern_ = ctx_ = dev_ = nullptr;
    c_bytes_ = 0;
}

bool npu_bfp16::open(const std::string & xclbin_path, std::string & err) {
    dev_ = xrtsh_device_open(0);
    if (!dev_) { err = shim_error("cannot open the NPU"); return false; }
    char name[256];
    if (xrtsh_device_name(dev_, name, sizeof(name)) > 0) device_name_ = name;

    ctx_ = xrtsh_hwctx_create(dev_, xclbin_path.c_str());
    if (!ctx_) { err = shim_error("cannot load " + xclbin_path); close(); return false; }
    kern_ = xrtsh_kernel_create_xclbin(ctx_, "MLIR_AIE");
    if (!kern_) { err = shim_error("cannot create the kernel"); close(); return false; }
    run_ = xrtsh_run_create(kern_);
    if (!run_) { err = shim_error("cannot create the run"); close(); return false; }
    return true;
}

int npu_bfp16::add_shape(const std::string & insts_path, int64_t M, int64_t K, int64_t N, std::string & err) {
    std::vector<uint8_t> instr;
    if (!read_file(insts_path, instr, err)) return -1;
    std::vector<uint32_t> words(instr.size() / 4);
    memcpy(words.data(), instr.data(), words.size() * 4);
    return add_shape(words, M, K, N, err);
}

int npu_bfp16::add_shape(const std::vector<uint32_t> & instr, int64_t M, int64_t K, int64_t N, std::string & err) {
    shape s;
    s.M = M; s.K = K; s.N = N;
    const size_t bytes = instr.size() * sizeof(uint32_t);
    s.n_words = (int) instr.size();
    s.ibo = xrtsh_bo_create_instr(dev_, kern_, bytes);
    if (!s.ibo) { err = shim_error("cannot allocate an instruction buffer"); return -1; }
    if (xrtsh_bo_write(s.ibo, instr.data(), bytes, 0) < 0 || xrtsh_bo_sync(s.ibo, 1) < 0) {
        err = shim_error("instruction upload");
        xrtsh_bo_free(s.ibo);
        return -1;
    }
    shapes_.push_back(s);
    return (int) shapes_.size() - 1;
}

int npu_bfp16::add_weights(const std::vector<uint8_t> & packed, std::string & err) {
    wbuf w;
    w.bytes = packed.size();
    w.bo = xrtsh_bo_create(dev_, w.bytes);
    if (!w.bo) { err = shim_error("cannot allocate a weight buffer"); return -1; }
    if (xrtsh_bo_write(w.bo, packed.data(), w.bytes, 0) < 0 || xrtsh_bo_sync(w.bo, 1) < 0) {
        err = shim_error("weight upload");
        xrtsh_bo_free(w.bo);
        return -1;
    }
    weights_.push_back(w);
    return (int) weights_.size() - 1;
}

int npu_bfp16::add_buffer(size_t bytes, std::string & err) {
    wbuf w;
    w.bytes = bytes;
    w.bo = xrtsh_bo_create(dev_, bytes);
    if (!w.bo) { err = shim_error("cannot allocate a buffer"); return -1; }
    uint8_t * p = (uint8_t *) xrtsh_bo_map(w.bo);
    if (!p) { err = shim_error("cannot map a buffer"); xrtsh_bo_free(w.bo); return -1; }
    memset(p, 0, bytes);
    if (xrtsh_bo_sync(w.bo, 1) < 0) { err = shim_error("buffer sync"); xrtsh_bo_free(w.bo); return -1; }
    weights_.push_back(w);
    return (int) weights_.size() - 1;
}

uint8_t * npu_bfp16::buffer_map(int h) { return (uint8_t *) xrtsh_bo_map(weights_.at(h).bo); }

bool npu_bfp16::buffer_sync(int h, size_t off, size_t n, std::string & err) {
    const auto t0 = std::chrono::steady_clock::now();
    if (xrtsh_bo_sync_range(weights_.at(h).bo, 1, n, off) < 0) { err = shim_error("buffer sync"); return false; }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    sync_in_ms += ms;
    sync_ms += ms;
    return true;
}

// Grows a data buffer to at least `need` bytes. Buffers only ever grow, so
// after the largest shape has run once nothing is allocated again.
bool npu_bfp16::ensure(void *& bo, size_t & have, size_t need, std::string & err) {
    if (bo && have >= need) return true;
    if (bo) xrtsh_bo_free(bo);
    bo = xrtsh_bo_create(dev_, need);
    if (!bo) { err = shim_error("cannot allocate a data buffer"); have = 0; return false; }
    have = need;
    return true;
}

bool npu_bfp16::set_a(int si, const std::vector<uint8_t> & packed, std::string & err) {
    const shape & s = shapes_.at(si);
    const size_t need = (size_t) (s.M * s.K / 8 * BFP16_BLOCK_BYTES);
    if (packed.size() != need) { err = "activation operand is the wrong size for this shape"; return false; }
    if (!a_slot(0, need, err)) return false;
    if (xrtsh_bo_write(abos_[0], packed.data(), need, 0) < 0 || xrtsh_bo_sync(abos_[0], 1) < 0) {
        err = shim_error("activation upload");
        return false;
    }
    return true;
}

bool npu_bfp16::run(int si, int wi, float * c, std::string & err, double * ms) {
    const shape & s = shapes_.at(si);
    if (!submit(s, 0, weights_.at(wi), err, ms)) return false;
    const size_t c_need = (size_t) (s.M * s.N) * sizeof(float);
    if (xrtsh_bo_sync(cbo_, 0) < 0 || xrtsh_bo_read(cbo_, c, c_need, 0) < 0) {
        err = shim_error("output read");
        return false;
    }
    return true;
}

uint8_t * npu_bfp16::a_slot(int slot, size_t bytes, std::string & err) {
    if ((int) abos_.size() <= slot) {
        abos_.resize(slot + 1, nullptr);
        a_caps_.resize(slot + 1, 0);
    }
    if (!ensure(abos_[slot], a_caps_[slot], bytes, err)) return nullptr;
    uint8_t * p = (uint8_t *) xrtsh_bo_map(abos_[slot]);
    if (!p) err = shim_error("cannot map the activation buffer");
    return p;
}

uint8_t * npu_bfp16::a_map(int si, std::string & err, int slot) {
    const shape & s = shapes_.at(si);
    return a_slot(slot, (size_t) (s.M * s.K / 8 * BFP16_BLOCK_BYTES), err);
}

bool npu_bfp16::run_mapped(int si, int wi, const float *& c, std::string & err, double * ms, int slot) {
    const shape & s = shapes_.at(si);
    // Only the bytes this shape uses: the buffers are sized for the largest
    // shape, and syncing all of them costs more than a small dispatch.
    const size_t a_n = (size_t) (s.M * s.K / 8 * BFP16_BLOCK_BYTES);
    if ((int) abos_.size() <= slot || a_caps_[slot] < a_n) { err = "no activations set for this shape"; return false; }
    auto t0 = std::chrono::steady_clock::now();
    if (xrtsh_bo_sync_range(abos_[slot], 1, a_n, 0) < 0) { err = shim_error("activation sync"); return false; }
    double took = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    sync_in_ms += took;
    sync_ms += took;
    if (!submit(s, slot, weights_.at(wi), err, ms)) return false;
    t0 = std::chrono::steady_clock::now();
    if (xrtsh_bo_sync_range(cbo_, 0, (size_t) (s.M * s.N) * sizeof(float), 0) < 0) { err = shim_error("output sync"); return false; }
    took = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    sync_out_ms += took;
    sync_ms += took;
    // The driver's sync from the device is a cache-flush loop with no fence
    // after it, so a load that follows can still see the stale line. Fence
    // before reading.
    _mm_mfence();
    c = (const float *) xrtsh_bo_map(cbo_);
    if (!c) { err = shim_error("cannot map the output buffer"); return false; }
    return true;
}

npu_bfp16::batch::~batch() {
    std::string ignore;
    if (in_flight_) wait(ignore);
    for (call & c : calls_) {
        if (c.run) xrtsh_run_free(c.run);
        for (void * bo : { c.a, c.b, c.c }) if (bo) xrtsh_bo_free(bo);
    }
}

// Mapped again every time a buffer may have changed: a buffer that grew is a
// new one, and its handle can equal the freed one's (the allocator reuses the
// address), so comparing handles would keep the old, unmapped pointer.
bool npu_bfp16::batch::map(call & c, std::string & err) {
    c.a_map = c.a ? (uint8_t *) xrtsh_bo_map(c.a) : nullptr;
    c.b_map = c.b ? (uint8_t *) xrtsh_bo_map(c.b) : nullptr;
    c.c_map = c.c ? (uint8_t *) xrtsh_bo_map(c.c) : nullptr;
    if ((c.a && !c.a_map) || (c.b && !c.b_map) || (c.c && !c.c_map)) { err = shim_error("cannot map a batch buffer"); return false; }
    return true;
}

bool npu_bfp16::batch::reserve(int n, size_t a_bytes, size_t b_bytes, size_t c_bytes, std::string & err) {
    if ((int) calls_.size() < n) calls_.resize(n);
    for (int i = 0; i < n; i++) {
        call & c = calls_[i];
        if (!npu_.ensure(c.a, c.a_cap, a_bytes, err) || !npu_.ensure(c.c, c.c_cap, c_bytes, err)) return false;
        if (b_bytes && !npu_.ensure(c.b, c.b_cap, b_bytes, err)) return false;
        if (!map(c, err)) return false;
    }
    return true;
}

bool npu_bfp16::batch::prepare(const std::vector<int> & shapes, std::string & err, const std::vector<int> & b) {
    if (in_flight_) { err = "batch prepared while running"; return false; }
    if (calls_.size() < shapes.size()) calls_.resize(shapes.size());
    n_ = shapes.size();
    if (!b.empty() && b.size() != 1 && b.size() != n_) { err = "one B buffer, or one per call"; return false; }
    b_.assign(n_, -1);
    for (size_t i = 0; i < n_ && !b.empty(); i++) b_[i] = b.size() == 1 ? b[0] : b[i];
    for (size_t i = 0; i < n_; i++) {
        call & c = calls_[i];
        const shape & s = npu_.shapes_.at(shapes[i]);
        const size_t a_need = (size_t) (s.M * s.K / 8 * BFP16_BLOCK_BYTES), b_need = (size_t) (s.N * s.K / 8 * BFP16_BLOCK_BYTES);
        const size_t c_need = (size_t) (s.M * s.N) * sizeof(float);
        const int bi = b_[i];
        if (!npu_.ensure(c.a, c.a_cap, a_need, err) || !npu_.ensure(c.c, c.c_cap, c_need, err)) return false;
        if (bi < 0 && !npu_.ensure(c.b, c.b_cap, b_need, err)) return false;
        if (bi >= 0 && npu_.weights_.at(bi).bytes < b_need) { err = "a B buffer too small for this shape"; return false; }
        if (!map(c, err)) return false;
        if (!c.run && !(c.run = xrtsh_run_create(npu_.kern_))) { err = shim_error("cannot create a batch run"); return false; }
        c.shape = shapes[i];
        xrtsh_run_set_arg_int(c.run, 0, 3);
        xrtsh_run_set_arg_bo (c.run, 1, s.ibo);
        xrtsh_run_set_arg_int(c.run, 2, s.n_words);
        xrtsh_run_set_arg_bo (c.run, 3, c.a);
        xrtsh_run_set_arg_bo (c.run, 4, bi >= 0 ? npu_.weights_[bi].bo : c.b);
        xrtsh_run_set_arg_bo (c.run, 5, c.c);
    }
    return true;
}

bool npu_bfp16::batch::start(std::string & err) {
    if (in_flight_) { err = "batch started twice"; return false; }
    const auto ts = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n_; i++) {
        const call & c = calls_[i];
        const shape & s = npu_.shapes_[c.shape];
        if (xrtsh_bo_sync_range(c.a, 1, (size_t) (s.M * s.K / 8 * BFP16_BLOCK_BYTES), 0) < 0 ||
            (b_[i] < 0 && xrtsh_bo_sync_range(c.b, 1, (size_t) (s.N * s.K / 8 * BFP16_BLOCK_BYTES), 0) < 0)) {
            err = shim_error("batch operand sync");
            return false;
        }
    }
    t_start_ = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t_start_ - ts).count();
    npu_.sync_in_ms += ms;
    npu_.sync_ms += ms;
    if (n_ == 1) {
        if (xrtsh_run_start(calls_[0].run) < 0) { err = shim_error("submit"); return false; }
    } else {
        rl_ = xrtsh_runlist_create(npu_.ctx_);
        if (!rl_) { err = shim_error("cannot create a runlist"); return false; }
        bool ok = true;
        for (size_t i = 0; i < n_ && ok; i++) ok = xrtsh_runlist_add(rl_, calls_[i].run) >= 0;
        if (!ok || xrtsh_runlist_execute(rl_) < 0) {
            err = shim_error("runlist");
            xrtsh_runlist_free(rl_);
            rl_ = nullptr;
            return false;
        }
    }
    in_flight_ = true;
    return true;
}

bool npu_bfp16::batch::wait(std::string & err, double * ms) {
    if (!in_flight_) { err = "batch waited without a start"; return false; }
    in_flight_ = false;
    bool ok;
    if (rl_) {
        ok = xrtsh_runlist_wait(rl_) >= 0;
        if (!ok) err = shim_error("runlist");
        xrtsh_runlist_free(rl_);
        rl_ = nullptr;
    } else {
        const int state = xrtsh_run_wait(calls_[0].run);
        ok = state == 4;
        if (!ok) err = "the kernel did not complete (state " + std::to_string(state) + "): " + xrtsh_last_error();
    }
    if (!ok) return false;
    const auto ts = std::chrono::steady_clock::now();
    if (ms) *ms = std::chrono::duration<double, std::milli>(ts - t_start_).count();
    for (size_t i = 0; i < n_; i++) {
        const shape & s = npu_.shapes_[calls_[i].shape];
        const size_t all = (size_t) (s.M * s.N) * sizeof(float);
        if (xrtsh_bo_sync_range(calls_[i].c, 0, c_prefix_ ? std::min(c_prefix_, all) : all, 0) < 0) {
            err = shim_error("batch output sync");
            return false;
        }
    }
    _mm_mfence();  // as in run_mapped
    const double sync = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
    npu_.sync_out_ms += sync;
    npu_.sync_ms += sync;
    return true;
}

bool npu_bfp16::submit(const shape & s, int slot, const wbuf & w, std::string & err, double * ms) {
    const size_t w_need = (size_t) (s.N * s.K / 8 * BFP16_BLOCK_BYTES);
    if (w.bytes != w_need) { err = "weights are the wrong size for this shape"; return false; }
    if ((int) abos_.size() <= slot || a_caps_[slot] < (size_t) (s.M * s.K / 8 * BFP16_BLOCK_BYTES)) {
        err = "no activations set for this shape";
        return false;
    }
    const size_t c_need = (size_t) (s.M * s.N) * sizeof(float);
    if (!ensure(cbo_, c_bytes_, c_need, err)) return false;

    // opcode, instruction buffer and length, then A, B, C as the design binds them
    xrtsh_run_set_arg_int(run_, 0, 3);
    xrtsh_run_set_arg_bo (run_, 1, s.ibo);
    xrtsh_run_set_arg_int(run_, 2, s.n_words);
    xrtsh_run_set_arg_bo (run_, 3, abos_[slot]);
    xrtsh_run_set_arg_bo (run_, 4, w.bo);
    xrtsh_run_set_arg_bo (run_, 5, cbo_);

    const auto t0 = std::chrono::steady_clock::now();
    if (xrtsh_run_start(run_) < 0) { err = shim_error("submit"); return false; }
    const int state = xrtsh_run_wait(run_);
    if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (state != 4) {
        err = "the kernel did not complete (state " + std::to_string(state) + "): " + xrtsh_last_error();
        return false;
    }
    return true;
}
