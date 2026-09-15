// The prefill/decode split lives entirely in supports_op and the buffer type.
// These are the two properties the scheduler reads to make that split, so they
// get checked directly rather than inferred from a model run.
//
// Traces: XDNA-HOST-BUFT, XDNA-BATCH-GATE, XDNA-WORK-FLOOR, XDNA-NO-OFFLOAD-OP

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>

static int failures = 0;

static void check(bool ok, const char * what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

// Builds a MUL_MAT of the given shape without allocating any data.
static ggml_tensor * make_mul_mat(ggml_context * ctx, ggml_type wtype, int64_t k, int64_t n, int64_t batch) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, wtype, k, n);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, batch);
    return ggml_mul_mat(ctx, a, b);
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("XDNA");
    if (dev == NULL) {
        fprintf(stderr, "XDNA device not registered - is GGML_BACKEND_PATH set?\n");
        return 2;
    }

    check(ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL,
          "device registers as ACCEL (llama.cpp puts ACCEL bufts first in the CPU list)");

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    check(ggml_backend_buft_is_host(buft),
          "buffer type is host-visible (lets the CPU backend read the same weights)");

    ggml_init_params ip = { ggml_tensor_overhead()*64, NULL, true };
    ggml_context * ctx = ggml_init(ip);

    // A weight big enough to clear the work floor at batch 512 but not at 1.
    const int64_t k = 2048, n = 2048;

    check(!ggml_backend_dev_supports_op(dev, make_mul_mat(ctx, GGML_TYPE_Q4_K, k, n, 1)),
          "batch 1 (decode) is not claimed");
    check(!ggml_backend_dev_supports_op(dev, make_mul_mat(ctx, GGML_TYPE_Q4_K, k, n, 31)),
          "batch 31 is not claimed (below the default threshold of 32)");
    check(ggml_backend_dev_supports_op(dev, make_mul_mat(ctx, GGML_TYPE_Q4_K, k, n, 512)),
          "batch 512 (prefill) is claimed");

    // 2*512*64*512 = 34 MFLOP, well under the 256 MFLOP floor (k stays a
    // multiple of the q4_K block size so ggml will build the tensor at all)
    check(!ggml_backend_dev_supports_op(dev, make_mul_mat(ctx, GGML_TYPE_Q4_K, 512, 64, 512)),
          "a tiny matmul is not claimed even at prefill batch");

    check(!ggml_backend_dev_supports_op(dev, make_mul_mat(ctx, GGML_TYPE_IQ2_XXS, k, n, 512)),
          "an unsupported quant type is not claimed");

    // offload_op is deliberately absent: that path copies the weight per op.
    check(!ggml_backend_dev_offload_op(dev, make_mul_mat(ctx, GGML_TYPE_Q4_K, k, n, 512)),
          "offload_op never claims an op (no per-op weight copies)");

    ggml_free(ctx);

    printf("%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
