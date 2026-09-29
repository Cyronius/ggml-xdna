// Real GGUF weights through our packer onto the NPU, checked on hardware.
//
// This is the whole weight path end to end, minus the kernel we do not have
// yet: read 4-bit weights out of a GGUF, rearrange them with our packer,
// rearrange activations with our tiler, dispatch one of openflowlm-next's
// built prefill GEMM kernels through the vendored driver wrapper, and compare
// what comes back against the same multiply done here in double precision.
//
// The reference uses the weights as the chunks hold them, with the scale
// narrowed to bfloat16, so what is measured here is the kernel's own error
// and not the narrowing. The narrowing is priced separately by
// test-q4-pack.
//
// The built kernel fixes the shape, so the weight matrix is assembled from
// whichever real tensors of the right K are needed to fill N rows. Which
// tensors those are does not matter; that they are real 4-bit GGUF weights
// packed by our own code does.
//
// Traces: HYBRID-NPU-GEMM

#include "npu_gemm.h"
#include "q4_pack.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#ifdef _MSC_VER
#define FSEEK64 _fseeki64
#else
#define FSEEK64 fseeko
#endif

namespace {

[[noreturn]] void die(const std::string & what) {
    fprintf(stderr, "error: %s\n", what.c_str());
    exit(1);
}

// Gather N rows of K 4-bit values from the file, taking whole tensors of the
// right row length in the order they appear until N rows are collected.
struct gathered {
    std::vector<uint8_t>     raw;    // n rows of row_bytes
    std::vector<std::string> from;   // which tensors contributed
    int    type = 0;
    size_t row_bytes = 0;
};

gathered gather_rows(const std::string & path, int64_t want_k, int64_t want_n) {
    ggml_context * gc = nullptr;
    gguf_init_params ip = { true, &gc };
    gguf_context * gg = gguf_init_from_file(path.c_str(), ip);
    if (!gg) die("cannot open " + path);
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) die("cannot open " + path);

    gathered g;
    int64_t have = 0;

    for (int64_t id = 0; id < gguf_get_n_tensors(gg) && have < want_n; id++) {
        const ggml_type ty = gguf_get_tensor_type(gg, id);
        if (ty != GGML_TYPE_Q4_0 && ty != GGML_TYPE_Q4_1) continue;
        if (g.type && (int) ty != g.type) continue;   // one type per matrix

        const char * name = gguf_get_tensor_name(gg, id);
        const ggml_tensor * meta = ggml_get_tensor(gc, name);
        if (ggml_n_dims(meta) != 2 || meta->ne[0] != want_k) continue;

        if (!g.type) {
            g.type = (int) ty;
            g.row_bytes = ggml_row_size(ty, want_k);
            g.raw.resize((size_t) want_n * g.row_bytes);
        }

        const int64_t take = std::min(meta->ne[1], want_n - have);
        FSEEK64(f, (int64_t) (gguf_get_data_offset(gg) + gguf_get_tensor_offset(gg, id)), SEEK_SET);
        if (fread(g.raw.data() + (size_t) have * g.row_bytes, 1, (size_t) take * g.row_bytes, f)
            != (size_t) take * g.row_bytes) die("short read on " + std::string(name));

        g.from.push_back(std::string(name) + " (" + std::to_string(take) + " rows)");
        have += take;
    }

    fclose(f);
    ggml_free(gc);
    gguf_free(gg);
    if (have < want_n) {
        die("this file has only " + std::to_string(have) + " 4-bit rows of length " +
            std::to_string(want_k) + "; need " + std::to_string(want_n));
    }
    return g;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, build_dir;
    int64_t N = 8192, K = 2048, T = 256;
    int     iters = 10, seed = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--build" && i + 1 < argc) build_dir = argv[++i];
        else if (a == "--n"     && i + 1 < argc) N = atoll(argv[++i]);
        else if (a == "--k"     && i + 1 < argc) K = atoll(argv[++i]);
        else if (a == "--t"     && i + 1 < argc) T = atoll(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--seed"  && i + 1 < argc) seed = atoi(argv[++i]);
        else if (model_path.empty())             model_path = a;
        else die("unexpected argument " + a);
    }
    if (model_path.empty() || build_dir.empty()) {
        die("usage: test-npu-gemm <model.gguf> --build <kernel-build-dir> [--n N --k K --t T] [--iters n]");
    }

    printf("model:  %s\n", model_path.c_str());
    printf("kernel: %s\n", build_dir.c_str());
    printf("shape:  N=%lld K=%lld T=%lld\n\n", (long long) N, (long long) K, (long long) T);

    // ---- weights: real 4-bit rows, packed by us -------------------------------
    const gathered g = gather_rows(model_path, K, N);
    printf("weights (%s) assembled from:\n", ggml_type_name((ggml_type) g.type));
    for (const std::string & s : g.from) printf("  %s\n", s.c_str());

    q4_pool_geometry geo;
    std::string err;
    if (!q4_pool_geometry_init(N, K, 2, geo, err)) die(err);

    std::vector<uint8_t> pool(geo.bytes());
    if (!q4_pack_pool(g.raw.data(), g.type, geo, pool.data(), err)) die(err);
    printf("packed into %lld chunks, %.2f MiB\n\n", (long long) geo.n_chunks, pool.size() / 1048576.0);

    // The weights as the chunks hold them. This is what the kernel sees, so
    // it is what the reference must multiply.
    std::vector<float> W((size_t) N * K);
    q4_dequant_pool(pool.data(), geo, W.data());

    // ---- activations: bfloat16, tiled by us ----------------------------------
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<uint16_t> x_tk((size_t) T * K);   // [token][feature], bfloat16
    for (auto & v : x_tk) v = q4_f32_to_bf16(dist(rng));

    std::vector<uint16_t> x_kt((size_t) K * T);   // transposed, as the tiler wants
    for (int64_t t = 0; t < T; t++) {
        for (int64_t k = 0; k < K; k++) x_kt[k * T + t] = x_tk[t * K + k];
    }
    std::vector<uint16_t> x_tiled((size_t) K * T);
    if (!npu_gemm_tile_activations(x_kt.data(), K, T, x_tiled.data(), err)) die(err);

    // ---- reference: the same multiply, in double -----------------------------
    std::vector<double> ref((size_t) N * T);
    for (int64_t n = 0; n < N; n++) {
        const float * w = W.data() + n * K;
        for (int64_t t = 0; t < T; t++) {
            double acc = 0.0;
            const uint16_t * x = x_tk.data() + t * K;
            for (int64_t k = 0; k < K; k++) acc += (double) w[k] * q4_bf16_to_f32(x[k]);
            ref[n * T + t] = acc;
        }
    }

    // ---- run it ---------------------------------------------------------------
    npu_gemm npu;
    if (!npu.open(build_dir, pool.size(), x_tiled.size() * 2, (size_t) N * T * 4, err)) die(err);
    printf("device: %s\n", npu.device_name().c_str());

    if (!npu.set_weights(pool.data(), pool.size(), err)) die(err);
    if (!npu.set_activations(x_tiled.data(), x_tiled.size() * 2, err)) die(err);

    double first_ms = 0, best_ms = 1e30, total_ms = 0;
    for (int i = 0; i < iters; i++) {
        double ms;
        if (!npu.run(ms, err)) die(err);
        if (i == 0) first_ms = ms;
        best_ms = std::min(best_ms, ms);
        total_ms += ms;
    }

    std::vector<float> y((size_t) N * T);
    if (!npu.get_output(y.data(), y.size() * 4, err)) die(err);

    // ---- compare ---------------------------------------------------------------
    double num = 0, den = 0, maxabs = 0, maxdiff = 0;
    bool finite = true;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = (double) y[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
        maxabs  = std::max(maxabs, std::fabs(ref[i]));
        maxdiff = std::max(maxdiff, std::fabs(d));
        if (!std::isfinite(y[i])) finite = false;
    }
    const double rel_fro = std::sqrt(num) / std::sqrt(std::max(den, 1e-300));
    const double maxrel  = maxdiff / std::max(maxabs, 1e-300);

    // Per-token agreement, which is what a prefill actually cares about.
    double worst_cos = 1.0;
    int64_t worst_tok = 0;
    for (int64_t t = 0; t < T; t++) {
        double dot = 0, na = 0, nb = 0;
        for (int64_t n = 0; n < N; n++) {
            const double a = y[n * T + t], b = ref[n * T + t];
            dot += a * b; na += a * a; nb += b * b;
        }
        const double c = dot / std::sqrt(std::max(na * nb, 1e-300));
        if (c < worst_cos) { worst_cos = c; worst_tok = t; }
    }

    const double gflop = 2.0 * (double) N * K * T / 1e9;
    printf("\nran %d times: first %.3f ms, best %.3f ms, mean %.3f ms\n", iters, first_ms, best_ms, total_ms / iters);
    printf("best is %.2f TFLOP/s and %.1f tokens/ms for this projection\n\n", gflop / best_ms, T / best_ms);

    printf("relative Frobenius error %.3e (the kernel's own gate is 5e-3)\n", rel_fro);
    printf("worst single element     %.3e relative\n", maxrel);
    printf("worst per-token cosine   %.9f (token %lld)\n\n", worst_cos, (long long) worst_tok);

    int failures = 0;
    auto check = [&](bool ok, const char * what) {
        printf("%s %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) failures++;
    };
    check(finite,             "every output value is finite");
    check(rel_fro <= 5e-3,    "the NPU result matches the reference within the kernel's gate");
    check(worst_cos > 0.9999, "every token's output points the same way as the reference");
    printf("%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
