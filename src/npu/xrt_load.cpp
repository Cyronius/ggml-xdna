#include "xrt_load.h"

#include "xdna-platform.h"
#include "xrt_shim.h"

#include <dlfcn.h>

#define XRTSH_FUNCS(X)                                                                                      \
    X(const char *, xrtsh_last_error, (void), ())                                                           \
    X(xrtsh_dev, xrtsh_device_open, (int a), (a))                                                           \
    X(int, xrtsh_device_name, (xrtsh_dev a, char * b, int c), (a, b, c))                                    \
    X(void, xrtsh_device_free, (xrtsh_dev a), (a))                                                          \
    X(xrtsh_ctx, xrtsh_hwctx_create, (xrtsh_dev a, const char * b), (a, b))                                 \
    X(void, xrtsh_hwctx_free, (xrtsh_ctx a), (a))                                                           \
    X(xrtsh_kernel, xrtsh_kernel_create, (xrtsh_ctx a, const char * b), (a, b))                             \
    X(xrtsh_kernel, xrtsh_kernel_create_xclbin, (xrtsh_ctx a, const char * b), (a, b))                      \
    X(xrtsh_bo, xrtsh_bo_create_instr, (xrtsh_dev a, xrtsh_kernel b, size_t c), (a, b, c))                  \
    X(void, xrtsh_kernel_free, (xrtsh_kernel a), (a))                                                       \
    X(xrtsh_bo, xrtsh_bo_create, (xrtsh_dev a, size_t b), (a, b))                                           \
    X(void *, xrtsh_bo_map, (xrtsh_bo a), (a))                                                              \
    X(int, xrtsh_bo_sync, (xrtsh_bo a, int b), (a, b))                                                      \
    X(int, xrtsh_bo_sync_range, (xrtsh_bo a, int b, size_t c, size_t d), (a, b, c, d))                      \
    X(int, xrtsh_bo_write, (xrtsh_bo a, const void * b, size_t c, size_t d), (a, b, c, d))                  \
    X(int, xrtsh_bo_read, (xrtsh_bo a, void * b, size_t c, size_t d), (a, b, c, d))                         \
    X(void, xrtsh_bo_free, (xrtsh_bo a), (a))                                                               \
    X(xrtsh_run, xrtsh_run_create, (xrtsh_kernel a), (a))                                                   \
    X(int, xrtsh_run_set_arg_int, (xrtsh_run a, int b, int c), (a, b, c))                                   \
    X(int, xrtsh_run_set_arg_bo, (xrtsh_run a, int b, xrtsh_bo c), (a, b, c))                               \
    X(xrtsh_bo, xrtsh_run_scratchpad_bo, (xrtsh_run a), (a))                                                \
    X(unsigned long long, xrtsh_bo_address, (xrtsh_bo a), (a))                                              \
    X(int, xrtsh_run_start, (xrtsh_run a), (a))                                                             \
    X(int, xrtsh_run_wait, (xrtsh_run a), (a))                                                              \
    X(int, xrtsh_run_wait_ms, (xrtsh_run a, unsigned b), (a, b))                                            \
    X(void, xrtsh_run_free, (xrtsh_run a), (a))                                                             \
    X(xrtsh_runlist, xrtsh_runlist_create, (xrtsh_ctx a), (a))                                              \
    X(int, xrtsh_runlist_add, (xrtsh_runlist a, xrtsh_run b), (a, b))                                       \
    X(int, xrtsh_runlist_execute, (xrtsh_runlist a), (a))                                                   \
    X(int, xrtsh_runlist_wait, (xrtsh_runlist a), (a))                                                      \
    X(void, xrtsh_runlist_free, (xrtsh_runlist a), (a))

namespace {
#define X(ret, name, params, args) ret(*p_##name) params = nullptr;
XRTSH_FUNCS(X)
#undef X
}  // namespace

// The wrapper's functions, by their usual names, calling into the loaded library.
extern "C" {
#define X(ret, name, params, args) \
    ret name params { return p_##name args; }
XRTSH_FUNCS(X)
#undef X
}

std::string xrt_load() {
    // global, so the wrapper's calls into XRT resolve against it; AMD's own packages install under /opt/xilinx
    void * xrt = dlopen("libxrt_coreutil.so.2", RTLD_NOW | RTLD_GLOBAL);
    if (!xrt) xrt = dlopen("/opt/xilinx/xrt/lib/libxrt_coreutil.so.2", RTLD_NOW | RTLD_GLOBAL);
    if (!xrt) return "no NPU driver (XRT's libxrt_coreutil.so.2 not found)";
    const std::string path = (xdna_module_dir() / "libggml-xdna-xrt.so").string();
    void * shim = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!shim) {
        const std::string e = dlerror();
        if (e.find("undefined symbol") != std::string::npos)
            return "the NPU driver's libxrt_coreutil.so.2 lacks a function this backend uses (" + e + ")";
        return "cannot load " + path + " (" + e + ")";
    }
#define X(ret, name, params, args) \
    if (!(p_##name = (ret(*) params) dlsym(shim, #name))) return path + " lacks " #name;
    XRTSH_FUNCS(X)
#undef X
    return {};
}
