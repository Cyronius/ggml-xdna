// Checks the XDNA backend's matmul against an fp64 reference built from the
// same weight bytes, and against what the CPU backend produces from those same
// bytes. Bit equality is the wrong bar - ggml's CPU kernel quantises the
// activation and we don't, so the two legitimately differ. What has to hold is
// that we are no further from the truth than the backend we displace.
//
// Same measure the open_kernels tests use, so numbers here are comparable to
// the ones in tools/open-kernels/.
//
// Traces: XDNA-MUL-MAT-AGREES

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct case_spec {
    const char * name;
    ggml_type    wtype;
    int64_t      k;       // reduction length
    int64_t      n;       // weight rows / output columns
    int64_t      m;       // batch
};

struct error_stats {
    double nrmse;
    double cos;
    double max_abs;
};

static error_stats compare(const std::vector<float> & got, const std::vector<double> & ref) {
    double se = 0.0, ss = 0.0, dot = 0.0, ng = 0.0, nr = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < got.size(); i++) {
        const double g = got[i];
        const double r = ref[i];
        se  += (g - r)*(g - r);
        ss  += r*r;
        dot += g*r;
        ng  += g*g;
        nr  += r*r;
        max_abs = std::max(max_abs, std::fabs(g - r));
    }
    return { std::sqrt(se / std::max(ss, 1e-300)),
             dot / std::max(std::sqrt(ng*nr), 1e-300),
             max_abs };
}

static std::vector<float> run_on(ggml_backend_t backend, const case_spec & c,
                                 const std::vector<uint8_t> & wdata, const std::vector<float> & bdata) {
    ggml_init_params ip = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, c.wtype, c.k, c.n);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k, c.m);
    ggml_tensor * d = ggml_mul_mat(ctx, a, b);

    ggml_backend_buffer_t buf =
        ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
    if (buf == NULL) {
        fprintf(stderr, "  alloc failed\n");
        ggml_free(ctx);
        return {};
    }

    ggml_backend_tensor_set(a, wdata.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, bdata.data(), 0, ggml_nbytes(b));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "  compute failed: %d\n", (int) st);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return {};
    }

    std::vector<float> out(c.n*c.m);
    ggml_backend_tensor_get(d, out.data(), 0, out.size()*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

// Dequantise the stored weights and accumulate in double. This is the truth
// both backends are approximating.
static std::vector<double> reference(const case_spec & c,
                                     const std::vector<uint8_t> & wdata, const std::vector<float> & bdata) {
    const ggml_type_traits * traits = ggml_get_type_traits(c.wtype);
    const size_t row_bytes = ggml_row_size(c.wtype, c.k);

    std::vector<double> out(c.n*c.m);
    std::vector<float>  row(c.k);

    for (int64_t i = 0; i < c.n; i++) {
        if (c.wtype == GGML_TYPE_F32) {
            memcpy(row.data(), wdata.data() + i*row_bytes, c.k*sizeof(float));
        } else {
            traits->to_float(wdata.data() + i*row_bytes, row.data(), c.k);
        }
        for (int64_t j = 0; j < c.m; j++) {
            const float * b = bdata.data() + j*c.k;
            double acc = 0.0;
            for (int64_t t = 0; t < c.k; t++) {
                acc += (double) row[t] * (double) b[t];
            }
            out[j*c.n + i] = acc;
        }
    }
    return out;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev_xdna = ggml_backend_dev_by_name("XDNA");
    if (dev_xdna == NULL) {
        fprintf(stderr, "XDNA device not registered - is GGML_BACKEND_PATH set?\n");
        return 2;
    }
    ggml_backend_dev_t dev_cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

    ggml_backend_t xdna = ggml_backend_dev_init(dev_xdna, NULL);
    ggml_backend_t cpu  = ggml_backend_dev_init(dev_cpu, NULL);

    const case_spec cases[] = {
        { "f32   256x128 x 64",  GGML_TYPE_F32,  256,  128,  64 },
        { "f16   256x128 x 64",  GGML_TYPE_F16,  256,  128,  64 },
        { "q8_0  512x256 x 64",  GGML_TYPE_Q8_0, 512,  256,  64 },
        { "q4_0  512x256 x 64",  GGML_TYPE_Q4_0, 512,  256,  64 },
        { "q4_K  512x256 x 64",  GGML_TYPE_Q4_K, 512,  256,  64 },
        { "q4_K 2048x512 x 512", GGML_TYPE_Q4_K, 2048, 512, 512 },
        { "q6_K  512x256 x 64",  GGML_TYPE_Q6_K, 512,  256,  64 },
    };

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    int failures = 0;

    printf("%-22s %-28s %-28s\n", "case", "XDNA vs fp64 ref", "CPU vs fp64 ref");

    for (const case_spec & c : cases) {
        std::vector<float> wf32(c.k*c.n);
        for (float & v : wf32) v = dist(rng);

        std::vector<uint8_t> wdata(ggml_row_size(c.wtype, c.k)*c.n);
        if (c.wtype == GGML_TYPE_F32) {
            memcpy(wdata.data(), wf32.data(), wdata.size());
        } else {
            ggml_quantize_chunk(c.wtype, wf32.data(), wdata.data(), 0, c.n, c.k, NULL);
        }

        std::vector<float> bdata(c.k*c.m);
        for (float & v : bdata) v = dist(rng);

        const std::vector<double> ref = reference(c, wdata, bdata);

        const std::vector<float> got_xdna = run_on(xdna, c, wdata, bdata);
        const std::vector<float> got_cpu  = run_on(cpu,  c, wdata, bdata);

        if (got_xdna.size() != ref.size() || got_cpu.size() != ref.size()) {
            printf("FAIL %-22s could not compute\n", c.name);
            failures++;
            continue;
        }

        const error_stats ex = compare(got_xdna, ref);
        const error_stats ec = compare(got_cpu,  ref);

        // We must be at least as close to the truth as the CPU kernel, with a
        // little slack for accumulation order, and correlated with it.
        const bool ok = ex.nrmse <= ec.nrmse*1.5 + 1e-6 && ex.cos > 0.9999;

        printf("%s %-22s nrmse %.2e cos %.6f   nrmse %.2e cos %.6f\n",
               ok ? "PASS" : "FAIL", c.name, ex.nrmse, ex.cos, ec.nrmse, ec.cos);
        if (!ok) failures++;
    }

    ggml_backend_free(xdna);
    ggml_backend_free(cpu);

    printf("%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
