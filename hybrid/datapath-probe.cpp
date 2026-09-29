// Step 0 of .claude/plans/backend-size-aware.md: how fast data moves between
// the Vulkan GPU's buffers and CPU memory on this machine, by each path ggml
// offers without patching anything. A backend next to Vulkan reads its inputs
// out of Vulkan's buffers and writes its results back, so these rates decide
// whether it can work.
//
//   datapath-probe.exe [--mb 4,16,64] [--iters 20]
//
// Paths, each checked for correct data and timed (median):
//   read, plain         ggml_backend_tensor_get into ordinary memory. On an
//                       integrated GPU the backend copies straight out of the
//                       mapped GPU buffer, which the CPU doesn't cache.
//   read, async         ggml_backend_tensor_get_async + synchronize into
//                       ordinary memory: the GPU copies into a staging buffer,
//                       then the CPU copies from there.
//   read, async pinned  the same into Vulkan's pinned host memory (its host
//                       buffer type): the GPU copies straight into it.
//   CPU reads pinned    memcpy out of that pinned memory, to see whether the
//                       CPU caches it.
//   write, plain        ggml_backend_tensor_set from ordinary memory.
//   write, async pinned ggml_backend_tensor_set_async from pinned memory.
//   synchronize, idle   ggml_backend_synchronize with nothing queued: the
//                       floor a handoff between backends pays.
// Run it again with GGML_VK_PREFER_HOST_MEMORY=1 to see what that setting
// changes; the first line says which way it ran.

#include "common.h"

#include "ggml-alloc.h"
#include "ggml.h"

#include <cstring>
#include <functional>
#include <sstream>

namespace {

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} // namespace

int main(int argc, char ** argv) {
    std::vector<int> mbs = { 4, 16, 64 };
    int iters = 20;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--mb" && i + 1 < argc) {
            mbs.clear();
            std::stringstream ss(argv[++i]);
            std::string item;
            while (std::getline(ss, item, ',')) mbs.push_back(atoi(item.c_str()));
        } else die("usage: datapath-probe [--mb 4,16,64] [--iters n]");
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = find_gpu();
    if (!dev) die("no GPU device registered");
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) die("cannot start the GPU backend");
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    if (!host_buft) die("the GPU device has no host buffer type");
    printf("device %s, GGML_VK_PREFER_HOST_MEMORY %s, %d runs per path, medians\n", ggml_backend_dev_name(dev),
           getenv("GGML_VK_PREFER_HOST_MEMORY") ? "set" : "not set", iters);
    printf("%6s  %13s %13s %13s %13s  %13s %13s  %s\n", "MB", "read plain", "read async", "read pinned", "CPU<-pinned",
           "write plain", "write pinned", "(GB/s)");

    for (int mb : mbs) {
        const size_t bytes = (size_t) mb << 20, n = bytes / sizeof(float);
        ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) n);
        ggml_backend_buffer_t dbuf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!dbuf) die("GPU buffer allocation failed");
        ggml_backend_buffer_t pbuf = ggml_backend_buft_alloc_buffer(host_buft, bytes);
        if (!pbuf) die("pinned buffer allocation failed");
        float * pinned = (float *) ggml_backend_buffer_get_base(pbuf);

        std::vector<float> src(n), dst(n);
        for (size_t i = 0; i < n; i++) src[i] = (float) (i % 100003) * 0.5f;
        ggml_backend_tensor_set(t, src.data(), 0, bytes);

        auto check = [&](const float * p, const char * what) {
            if (memcmp(p, src.data(), bytes) != 0) die(std::string("wrong data after ") + what);
        };
        auto rate = [&](const std::function<void()> & fn, const std::function<void()> & after) {
            std::vector<double> ms;
            for (int i = 0; i < iters + 2; i++) {
                const clk::time_point t0 = clk::now();
                fn();
                const double d = ms_since(t0);
                if (after) after();
                if (i >= 2) ms.push_back(d);
            }
            return bytes / (median(ms) * 1e6);
        };

        const double r_plain = rate([&] { ggml_backend_tensor_get(t, dst.data(), 0, bytes); },
                                    [&] { check(dst.data(), "read plain"); std::fill(dst.begin(), dst.end(), 0.f); });
        const double r_async = rate(
            [&] {
                ggml_backend_tensor_get_async(be, t, dst.data(), 0, bytes);
                ggml_backend_synchronize(be);
            },
            [&] { check(dst.data(), "read async"); std::fill(dst.begin(), dst.end(), 0.f); });
        const double r_pinned = rate(
            [&] {
                ggml_backend_tensor_get_async(be, t, pinned, 0, bytes);
                ggml_backend_synchronize(be);
            },
            [&] { check(pinned, "read pinned"); });
        const double cpu_pinned = rate([&] { memcpy(dst.data(), pinned, bytes); }, [&] { check(dst.data(), "CPU read of pinned"); });

        // writes: the source is the pattern, so the check reads the GPU buffer
        // back through the fast path
        auto check_gpu = [&](const char * what) {
            std::fill(dst.begin(), dst.end(), 0.f);
            ggml_backend_tensor_get_async(be, t, dst.data(), 0, bytes);
            ggml_backend_synchronize(be);
            check(dst.data(), what);
        };
        const double w_plain = rate([&] { ggml_backend_tensor_set(t, src.data(), 0, bytes); }, [&] { check_gpu("write plain"); });
        memcpy(pinned, src.data(), bytes);
        const double w_pinned = rate(
            [&] {
                ggml_backend_tensor_set_async(be, t, pinned, 0, bytes);
                ggml_backend_synchronize(be);
            },
            [&] { check_gpu("write pinned"); });

        printf("%6d  %13.2f %13.2f %13.2f %13.2f  %13.2f %13.2f\n", mb, r_plain, r_async, r_pinned, cpu_pinned, w_plain,
               w_pinned);
        ggml_backend_buffer_free(pbuf);
        ggml_backend_buffer_free(dbuf);
        ggml_free(ctx);
    }

    std::vector<double> us;
    for (int i = 0; i < 1000; i++) {
        const clk::time_point t0 = clk::now();
        ggml_backend_synchronize(be);
        us.push_back(ms_since(t0) * 1000.0);
    }
    printf("synchronize with nothing queued: median %.1f us\n", median(us));
    ggml_backend_free(be);
    return 0;
}
