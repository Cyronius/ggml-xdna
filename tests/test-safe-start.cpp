// When the NPU can't run the backend, llama.cpp must not see XDNA0 at all:
// the backend then changes nothing, where offering the device would have it
// claim work it can't do. Simulated with a kernel path that names nothing,
// which fails the same check a missing driver or an untested chip does.
// Host-only mode (GGML_XDNA_HOST_ONLY) must still offer the device, for
// tests on machines without an NPU; it's cleared here so it can't hide the
// failure.
//
// --no-driver, on a machine without the NPU driver (CI): keeps the real
// kernel, so the driver check is the one that fails. The backend must still
// load, although the driver's DLL it links is missing.
//
// Traces: XDNA-SAFE-START

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 2 || strcmp(argv[1], "--no-driver") != 0) _putenv("GGML_XDNA_KERNELS=Z:\\no\\such\\kernel.xclbin");
    _putenv("GGML_XDNA_HOST_ONLY=");
    ggml_backend_load_all();

    int failures = 0;
    auto check = [&](bool ok, const char * what) {
        printf("%s %s\n", ok ? "PASS" : "FAIL", what);
        failures += !ok;
    };
    check(ggml_backend_reg_by_name("XDNA") != nullptr, "the backend loads (GGML_BACKEND_PATH set)");
    check(ggml_backend_dev_by_name("XDNA0") == nullptr, "no XDNA0 device when the NPU can't run the kernel");
    printf("%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
