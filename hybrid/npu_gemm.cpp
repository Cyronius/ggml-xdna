#include "npu_gemm.h"

#include "xrt_shim.h"

#include <chrono>
#include <cstdio>
#include <cstring>

namespace {

std::string shim_error(const std::string & what) {
    return what + ": " + xrtsh_last_error();
}

bool read_file(const std::string & path, std::vector<uint8_t> & out, std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(n);
    const bool ok = fread(out.data(), 1, n, f) == (size_t) n;
    fclose(f);
    if (!ok) { err = "short read on " + path; return false; }
    return true;
}

} // namespace

bool npu_gemm_tile_activations(const uint16_t * x_kt, int64_t K, int64_t T,
                               uint16_t * out, std::string & err) {
    if (K % NPU_GEMM_K_TILE != 0 || T % NPU_GEMM_TILE_N != 0) {
        err = "activations [" + std::to_string(K) + "," + std::to_string(T) +
              "] do not tile into (" + std::to_string(NPU_GEMM_K_TILE) + "," +
              std::to_string(NPU_GEMM_TILE_N) + ")";
        return false;
    }
    const int64_t n_kb = K / NPU_GEMM_K_TILE;
    const int64_t n_nb = T / NPU_GEMM_TILE_N;
    const int64_t n_ks = NPU_GEMM_K_TILE / NPU_GEMM_MAC_S;   // 8
    const int64_t n_nt = NPU_GEMM_TILE_N / NPU_GEMM_MAC_T;   // 4

    // Tiles in (k-block, n-block) order, each tile internally (s, t).
    size_t o = 0;
    for (int64_t kb = 0; kb < n_kb; kb++) {
        for (int64_t nb = 0; nb < n_nb; nb++) {
            for (int64_t ks = 0; ks < n_ks; ks++) {
                for (int64_t nt = 0; nt < n_nt; nt++) {
                    for (int64_t s = 0; s < NPU_GEMM_MAC_S; s++) {
                        const int64_t k = kb * NPU_GEMM_K_TILE + ks * NPU_GEMM_MAC_S + s;
                        const uint16_t * src = x_kt + k * T + nb * NPU_GEMM_TILE_N + nt * NPU_GEMM_MAC_T;
                        std::memcpy(out + o, src, NPU_GEMM_MAC_T * sizeof(uint16_t));
                        o += NPU_GEMM_MAC_T;
                    }
                }
            }
        }
    }
    return true;
}

npu_gemm::~npu_gemm() { close(); }

void npu_gemm::close() {
    if (run_)  xrtsh_run_free(run_);
    if (ybo_)  xrtsh_bo_free(ybo_);
    if (xbo_)  xrtsh_bo_free(xbo_);
    if (wbo_)  xrtsh_bo_free(wbo_);
    if (ibo_)  xrtsh_bo_free(ibo_);
    if (kern_) xrtsh_kernel_free(kern_);
    if (ctx_)  xrtsh_hwctx_free(ctx_);
    if (dev_)  xrtsh_device_free(dev_);
    run_ = ybo_ = xbo_ = wbo_ = ibo_ = kern_ = ctx_ = dev_ = nullptr;
}

bool npu_gemm::open(const std::string & build_dir, size_t w_bytes, size_t x_bytes, size_t y_bytes, std::string & err) {
    dev_ = xrtsh_device_open(0);
    if (!dev_) { err = shim_error("cannot open the NPU"); return false; }
    char name[256];
    if (xrtsh_device_name(dev_, name, sizeof(name)) > 0) device_name_ = name;

    ctx_ = xrtsh_hwctx_create(dev_, (build_dir + "/final.xclbin").c_str());
    if (!ctx_) { err = shim_error("cannot load " + build_dir + "/final.xclbin"); close(); return false; }

    kern_ = xrtsh_kernel_create_xclbin(ctx_, "MLIR_AIE");
    if (!kern_) { err = shim_error("cannot create the kernel"); close(); return false; }

    std::vector<uint8_t> instr;
    if (!read_file(build_dir + "/insts.bin", instr, err)) { close(); return false; }
    n_instr_words_ = (int) (instr.size() / 4);

    ibo_ = xrtsh_bo_create_instr(dev_, kern_, instr.size());
    if (!ibo_) { err = shim_error("cannot allocate the instruction buffer"); close(); return false; }
    xrtsh_bo_write(ibo_, instr.data(), instr.size(), 0);
    xrtsh_bo_sync(ibo_, 1);

    wbo_ = xrtsh_bo_create(dev_, w_bytes);
    xbo_ = xrtsh_bo_create(dev_, x_bytes);
    ybo_ = xrtsh_bo_create(dev_, y_bytes);
    if (!wbo_ || !xbo_ || !ybo_) { err = shim_error("cannot allocate the data buffers"); close(); return false; }

    // Argument order is the shim's convention for a classic kernel: opcode,
    // the instruction buffer, its length, then the data buffers as the
    // kernel binds them.
    run_ = xrtsh_run_create(kern_);
    if (!run_) { err = shim_error("cannot create the run"); close(); return false; }
    xrtsh_run_set_arg_int(run_, 0, 3);
    xrtsh_run_set_arg_bo (run_, 1, ibo_);
    xrtsh_run_set_arg_int(run_, 2, n_instr_words_);
    xrtsh_run_set_arg_bo (run_, 3, wbo_);
    xrtsh_run_set_arg_bo (run_, 4, xbo_);
    xrtsh_run_set_arg_bo (run_, 5, ybo_);
    return true;
}

bool npu_gemm::set_weights(const void * data, size_t n, std::string & err) {
    if (xrtsh_bo_write(wbo_, data, n, 0) < 0) { err = shim_error("weight upload"); return false; }
    if (xrtsh_bo_sync(wbo_, 1) < 0)           { err = shim_error("weight sync");   return false; }
    return true;
}

bool npu_gemm::set_activations(const void * data, size_t n, std::string & err) {
    if (xrtsh_bo_write(xbo_, data, n, 0) < 0) { err = shim_error("activation upload"); return false; }
    if (xrtsh_bo_sync(xbo_, 1) < 0)           { err = shim_error("activation sync");   return false; }
    return true;
}

bool npu_gemm::run(double & ms, std::string & err) {
    const auto t0 = std::chrono::steady_clock::now();
    if (xrtsh_run_start(run_) < 0) { err = shim_error("submit"); return false; }
    const int state = xrtsh_run_wait(run_);
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (state != 4) {
        err = "the kernel did not complete (state " + std::to_string(state) + "): " + xrtsh_last_error();
        return false;
    }
    return true;
}

bool npu_gemm::get_output(void * data, size_t n, std::string & err) {
    if (xrtsh_bo_sync(ybo_, 0) < 0)          { err = shim_error("output sync"); return false; }
    if (xrtsh_bo_read(ybo_, data, n, 0) < 0) { err = shim_error("output read"); return false; }
    return true;
}
