// The host's small steps between the NPU's matmuls, against ggml's CPU backend; host-only mode runs them, no NPU needed.
// Traces: XDNA-HOST-OPS

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

struct op_case {
    const char * name;
    double tol;  // normalised RMS error allowed against the CPU
    std::function<ggml_tensor *(ggml_context *, std::vector<ggml_tensor *> &)> build;  // records its inputs
};

static ggml_tensor * input(ggml_context * ctx, std::vector<ggml_tensor *> & in, ggml_type type, int64_t a, int64_t b = 1,
                           int64_t c = 1, const char * name = "x") {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, type, a, b, c);
    ggml_set_name(t, name);
    in.push_back(t);
    return t;
}

static std::vector<float> run_on(ggml_backend_t backend, const op_case & c) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> in;
    ggml_tensor * out = c.build(ctx, in);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return {};
    }
    // the same inputs on every backend
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.5f);
    std::uniform_real_distribution<float> factor(1.0f, 4.0f);
    for (ggml_tensor * t : in) {
        const int64_t n = ggml_nelements(t);
        if (t->type == GGML_TYPE_I32) {
            std::vector<int32_t> pos((size_t) n);
            for (int64_t i = 0; i < n; i++) pos[(size_t) i] = 5000 + 300 * (int32_t) i;  // past a short context
            ggml_backend_tensor_set(t, pos.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<float> v((size_t) n);
            for (float & x : v) x = strcmp(t->name, "factors") == 0 ? factor(rng) : normal(rng);
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    std::vector<float> result;
    if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
        result.resize((size_t) ggml_nelements(out));
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return result;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev_xdna = ggml_backend_dev_by_name("XDNA0");
    if (!dev_xdna) {
        fprintf(stderr, "XDNA device not registered - is GGML_BACKEND_PATH set?\n");
        return 2;
    }
    ggml_backend_t xdna = ggml_backend_dev_init(dev_xdna, NULL);
    ggml_backend_t cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), NULL);

    const int64_t D = 256, T = 8, HD = 64, H = 4;
    const double exact = 1e-5;
    const double gelu_table = 3e-3;  // ggml-cpu's GELUs read a 16-bit table; the GPU's and ours compute them
    using in_t = std::vector<ggml_tensor *>;
    const auto rope = [&](int n_dims, int mode, int n_ctx_orig, float freq_scale, float ext_factor, bool factors) {
        return [=](ggml_context * ctx, in_t & in) {
            ggml_tensor * x = input(ctx, in, GGML_TYPE_F32, HD, H, T);
            ggml_tensor * pos = input(ctx, in, GGML_TYPE_I32, T, 1, 1, "pos");
            ggml_tensor * ff = factors ? input(ctx, in, GGML_TYPE_F32, n_dims / 2, 1, 1, "factors") : nullptr;
            return ggml_rope_ext(ctx, x, pos, ff, n_dims, mode, n_ctx_orig, 10000.0f, freq_scale, ext_factor, 1.0f,
                                 32.0f, 1.0f);
        };
    };
    const op_case cases[] = {
        { "rms_norm", exact,
          [&](ggml_context * ctx, in_t & in) { return ggml_rms_norm(ctx, input(ctx, in, GGML_TYPE_F32, D, T), 1e-6f); } },
        { "mul by a scale row", exact,
          [&](ggml_context * ctx, in_t & in) {
              ggml_tensor * x = input(ctx, in, GGML_TYPE_F32, D, T);
              return ggml_mul(ctx, x, input(ctx, in, GGML_TYPE_F32, D));
          } },
        { "add", exact,
          [&](ggml_context * ctx, in_t & in) {
              ggml_tensor * x = input(ctx, in, GGML_TYPE_F32, D, T);
              return ggml_add(ctx, x, input(ctx, in, GGML_TYPE_F32, D, T));
          } },
        { "add a bias row", exact,
          [&](ggml_context * ctx, in_t & in) {
              ggml_tensor * x = input(ctx, in, GGML_TYPE_F32, D, T);
              return ggml_add(ctx, x, input(ctx, in, GGML_TYPE_F32, D));
          } },
        { "scale and bias", exact,
          [&](ggml_context * ctx, in_t & in) {
              return ggml_scale_bias(ctx, input(ctx, in, GGML_TYPE_F32, D, T), 0.37f, 0.25f);
          } },
        { "swiglu, split", exact,
          [&](ggml_context * ctx, in_t & in) {
              ggml_tensor * g = input(ctx, in, GGML_TYPE_F32, D, T);
              return ggml_swiglu_split(ctx, g, input(ctx, in, GGML_TYPE_F32, D, T));
          } },
        { "geglu, split", gelu_table,
          [&](ggml_context * ctx, in_t & in) {
              ggml_tensor * g = input(ctx, in, GGML_TYPE_F32, D, T);
              return ggml_geglu_split(ctx, g, input(ctx, in, GGML_TYPE_F32, D, T));
          } },
        { "geglu_erf, split", exact,
          [&](ggml_context * ctx, in_t & in) {
              ggml_tensor * g = input(ctx, in, GGML_TYPE_F32, D, T);
              return ggml_geglu_erf_split(ctx, g, input(ctx, in, GGML_TYPE_F32, D, T));
          } },
        { "geglu_quick, split", gelu_table,
          [&](ggml_context * ctx, in_t & in) {
              ggml_tensor * g = input(ctx, in, GGML_TYPE_F32, D, T);
              return ggml_geglu_quick_split(ctx, g, input(ctx, in, GGML_TYPE_F32, D, T));
          } },
        { "reglu, split", exact,
          [&](ggml_context * ctx, in_t & in) {
              ggml_tensor * g = input(ctx, in, GGML_TYPE_F32, D, T);
              return ggml_reglu_split(ctx, g, input(ctx, in, GGML_TYPE_F32, D, T));
          } },
        { "swiglu, one tensor", exact,
          [&](ggml_context * ctx, in_t & in) { return ggml_swiglu(ctx, input(ctx, in, GGML_TYPE_F32, 2 * D, T)); } },
        { "swiglu, one tensor, swapped", exact,
          [&](ggml_context * ctx, in_t & in) { return ggml_swiglu_swapped(ctx, input(ctx, in, GGML_TYPE_F32, 2 * D, T)); } },
        { "geglu, one tensor", gelu_table,
          [&](ggml_context * ctx, in_t & in) { return ggml_geglu(ctx, input(ctx, in, GGML_TYPE_F32, 2 * D, T)); } },
        { "geglu_erf, one tensor", exact,
          [&](ggml_context * ctx, in_t & in) { return ggml_geglu_erf(ctx, input(ctx, in, GGML_TYPE_F32, 2 * D, T)); } },
        { "rope, normal", exact, rope(HD, GGML_ROPE_TYPE_NORMAL, 0, 1.0f, 0.0f, false) },
        { "rope, neox", exact, rope(HD, GGML_ROPE_TYPE_NEOX, 0, 1.0f, 0.0f, false) },
        { "rope, normal, half the head", exact, rope(HD / 2, GGML_ROPE_TYPE_NORMAL, 0, 1.0f, 0.0f, false) },
        { "rope, neox, three quarters", exact, rope(HD * 3 / 4, GGML_ROPE_TYPE_NEOX, 0, 1.0f, 0.0f, false) },
        { "rope, neox, yarn", exact, rope(HD, GGML_ROPE_TYPE_NEOX, 4096, 0.25f, 1.0f, false) },
        { "rope, neox, frequency factors", exact, rope(HD, GGML_ROPE_TYPE_NEOX, 4096, 1.0f, 0.0f, true) },
        { "rope, normal, yarn and factors", exact, rope(HD, GGML_ROPE_TYPE_NORMAL, 4096, 0.25f, 1.0f, true) },
    };

    int failures = 0;
    for (const op_case & c : cases) {
        const std::vector<float> got = run_on(xdna, c), want = run_on(cpu, c);
        if (got.empty() || got.size() != want.size()) {
            printf("FAIL %-32s could not compute\n", c.name);
            failures++;
            continue;
        }
        double se = 0, ss = 0;
        for (size_t i = 0; i < got.size(); i++) {
            se += ((double) got[i] - want[i]) * ((double) got[i] - want[i]);
            ss += (double) want[i] * want[i];
        }
        const double nrmse = std::sqrt(se / std::max(ss, 1e-300));
        const bool ok = std::isfinite(nrmse) && nrmse <= c.tol;
        printf("%s %-32s nrmse %.2e (allowed %.0e)\n", ok ? "PASS" : "FAIL", c.name, nrmse, c.tol);
        failures += !ok;
    }

    ggml_backend_free(xdna);
    ggml_backend_free(cpu);
    printf("%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
