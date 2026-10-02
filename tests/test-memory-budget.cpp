// The NPU's weight copies are held to a memory limit, decided through
// supports_op like the rest of the policy. Run with GGML_XDNA_MAX_COPY_GB=0.05
// (51.2 MiB) and the other settings at their defaults.
//
// A 2048 x 2048 weight's copy is 4.5 MiB, so a layer of four is 18 MiB:
// layers 0 and 1 fit (36 MiB), layer 2's fourth weight would pass the limit.
//
// Traces: XDNA-MEMORY-BUDGET

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <string>

static int failures = 0;

static void check(bool ok, const char * what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("XDNA0");
    ggml_backend_dev_t vk = ggml_backend_dev_by_name("Vulkan0");
    if (dev == NULL || vk == NULL) {
        fprintf(stderr, "XDNA0 or Vulkan0 not registered - is GGML_BACKEND_PATH set?\n");
        return 2;
    }

    // weights as llama.cpp loads them: named by layer, in a Vulkan buffer marked as weights
    ggml_init_params wp = { ggml_tensor_overhead() * 32, NULL, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * w[4][4] = {};
    for (int l = 0; l < 3; l++)
        for (int i = 0; i < 4; i++) {
            w[l][i] = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_0, 2048, 2048);
            ggml_set_name(w[l][i], ("blk." + std::to_string(l) + ".w" + std::to_string(i) + ".weight").c_str());
        }
    ggml_tensor * small = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_0, 2048, 512);  // 1.1 MiB
    ggml_set_name(small, "blk.3.w0.weight");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, ggml_backend_dev_buffer_type(vk));
    if (buf == NULL) {
        fprintf(stderr, "cannot allocate the weights in Vulkan memory\n");
        return 2;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_init_params ip = { ggml_tensor_overhead() * 256, NULL, true };
    ggml_context * ctx = ggml_init(ip);
    auto claimed = [&](ggml_tensor * weight) {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, weight->ne[0], 512);
        return ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, weight, x));
    };

    check(claimed(small), "with no context started, the limit doesn't apply (llama.cpp's load-time checks)");

    ggml_backend_t be = ggml_backend_dev_init(dev, NULL);
    if (be == NULL) {
        fprintf(stderr, "cannot start the XDNA0 backend\n");
        return 2;
    }

    // a 72 MiB weight with no data, as in llama.cpp's memory-fitting trial
    ggml_tensor * trial = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 2048, 32768);
    ggml_set_name(trial, "blk.0.trial.weight");
    check(claimed(trial), "a weight with no data is claimed, and doesn't count against the limit");

    bool all = true;
    for (int l = 0; l < 2; l++)
        for (int i = 0; i < 4; i++) all &= claimed(w[l][i]);
    check(all, "layers 0 and 1 fit (36 MiB of 51.2) and are claimed");
    check(claimed(w[2][0]) && claimed(w[2][1]) && claimed(w[2][2]), "layer 2's first three weights fit when asked");
    check(!claimed(w[2][3]), "layer 2's fourth weight would pass the limit: not claimed");
    check(!claimed(w[2][0]) && !claimed(w[2][1]) && !claimed(w[2][2]),
          "layer 2's first three then stay on the GPU too (no layer is split)");
    check(!claimed(small), "a weight that would still fit is not claimed once the limit is reached");
    check(claimed(w[0][0]) && claimed(w[1][3]), "what was taken stays taken");

    ggml_backend_free(be);
    be = ggml_backend_dev_init(dev, NULL);
    check(claimed(small) && claimed(w[2][3]), "after the context is freed, the next one starts with nothing counted");
    ggml_backend_free(be);

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_free(wctx);

    printf("%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
