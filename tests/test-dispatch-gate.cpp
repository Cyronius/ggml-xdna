// What the backend takes from Vulkan lives entirely in supports_op, the
// device type and the buffer type: the three things the scheduler reads to
// decide. So they're checked directly rather than inferred from a model run.
// Run with GGML_XDNA_MIN_BATCH and GGML_XDNA_MIN_MFLOP unset (the defaults).
//
// Traces: XDNA-SHARED-BUFT, XDNA-BATCH-GATE, XDNA-WORK-FLOOR, XDNA-NO-OFFLOAD-OP, XDNA-SIZE-POLICY

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>

static int failures = 0;

static void check(bool ok, const char * what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

// A MUL_MAT of the given shape, without allocating any data.
static ggml_tensor * make_mul_mat(ggml_context * ctx, ggml_type wtype, int64_t k, int64_t n, int64_t batch) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, wtype, k, n);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, batch);
    return ggml_mul_mat(ctx, a, b);
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("XDNA0");
    if (dev == NULL) {
        fprintf(stderr, "XDNA0 device not registered - is GGML_BACKEND_PATH set?\n");
        return 2;
    }
    ggml_backend_dev_t vk = ggml_backend_dev_by_name("Vulkan0");
    if (vk == NULL) {
        fprintf(stderr, "Vulkan0 not registered - the backend works next to it\n");
        return 2;
    }

    check(ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU,
          "device registers as an integrated GPU (llama.cpp ranks accelerators below every GPU)");
    check(ggml_backend_dev_buffer_type(dev) == ggml_backend_dev_buffer_type(vk),
          "its buffer type is Vulkan's (ops change hands with no copies)");
    check(ggml_backend_dev_supports_buft(dev, ggml_backend_dev_buffer_type(vk)),
          "it can read Vulkan's buffers");
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    check(cpu && ggml_backend_dev_supports_buft(dev, ggml_backend_dev_buffer_type(cpu)),
          "it can read host buffers (so its pieces never need copied inputs, which skip waiting for Vulkan)");
    size_t free = 1, total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    check(free == 0 && total > 0, "reports no free memory, so llama.cpp gives it no layers");

    ggml_init_params ip = { ggml_tensor_overhead() * 64, NULL, true };
    ggml_context * ctx = ggml_init(ip);
    const int64_t k = 2048, n = 2048;

    check(!ggml_backend_dev_supports_op(dev, make_mul_mat(ctx, GGML_TYPE_Q4_K, k, n, 1)),
          "batch 1 (reply generation) is not claimed");
    check(!ggml_backend_dev_supports_op(dev, make_mul_mat(ctx, GGML_TYPE_Q4_K, k, n, 1023)),
          "batch 1023 is not claimed (below the default threshold of 1024)");
    check(ggml_backend_dev_supports_op(dev, make_mul_mat(ctx, GGML_TYPE_Q4_K, k, n, 1024)),
          "batch 1024, q4_K 2048x2048 is claimed");
    check(ggml_backend_dev_supports_op(dev, make_mul_mat(ctx, GGML_TYPE_Q4_0, k, n, 2048)),
          "batch 2048, q4_0 is claimed");

    // 2*1024*64*1024 = 134 MFLOP, under the 256 MFLOP floor
    check(!ggml_backend_dev_supports_op(dev, make_mul_mat(ctx, GGML_TYPE_Q4_K, 1024, 64, 1024)),
          "a tiny matmul is not claimed even at a large batch");

    // attention's matmuls read keys from the cache through views, not weights
    {
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, 4 * n);
        ggml_tensor * view = ggml_view_2d(ctx, cache, k, n, cache->nb[1], 0);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1024);
        check(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, view, b)),
              "a matmul whose left side is a view (a cache read, not a weight) is not claimed");
        ggml_tensor * computed = ggml_add(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n),
                                          ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n));
        check(!ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, computed, b)),
              "a matmul whose left side is computed (not a weight) is not claimed");
        check(!ggml_backend_dev_supports_op(dev, computed), "other ops (ADD) are not claimed");
    }

    // offload_op is deliberately absent: that path copies the weight per op.
    check(!ggml_backend_dev_offload_op(dev, make_mul_mat(ctx, GGML_TYPE_Q4_K, k, n, 2048)),
          "offload_op never claims an op (no per-op weight copies)");

    ggml_free(ctx);

    printf("%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
