// Stage 3b of .claude/plans/npu-prefill-beats-gpu.md: real Qwen3 weights,
// packed by hybrid/bfp16_pack.cpp, through the bfp16 kernel on the NPU via
// hybrid/npu_bfp16.cpp, for every projection of one layer.
//
// Two comparisons per projection:
//   exact  against a double-precision multiply of exactly the bfp16 values the
//          NPU was given. With float32 running sums and float32 output this is
//          the kernel's own error, and should be at float32 rounding.
//   bfp16  against the multiply of the unrounded values. This is what the
//          8-bit format costs, which HYBRID-BFP16-NUMERICS prices for the whole
//          model.
// Activations are random with a few large outliers per row. Real ones come in
// stage 3c.
//
// One hardware context per xclbin. Until the kernel takes K at run time, the
// K = 6144 projection (down) needs its own xclbin and so its own context.
//
// Traces: HYBRID-NPU-GEMM

#include "bfp16_pack.h"
#include "npu_bfp16.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
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

// One tensor from the file as float32, rows of ne0.
std::vector<float> load_tensor(const std::string & path, const char * name, int64_t & rows, int64_t & cols) {
    ggml_context * gc = nullptr;
    gguf_init_params ip = { true, &gc };
    gguf_context * gg = gguf_init_from_file(path.c_str(), ip);
    if (!gg) die("cannot read " + path);
    const int64_t id = gguf_find_tensor(gg, name);
    if (id < 0) die(std::string("no tensor ") + name);
    const ggml_tensor * t = ggml_get_tensor(gc, name);
    cols = t->ne[0];
    rows = t->ne[1];
    const ggml_type ty = gguf_get_tensor_type(gg, id);
    std::vector<uint8_t> raw(gguf_get_tensor_size(gg, id));
    FILE * f = fopen(path.c_str(), "rb");
    FSEEK64(f, (int64_t) (gguf_get_data_offset(gg) + gguf_get_tensor_offset(gg, id)), SEEK_SET);
    if (fread(raw.data(), 1, raw.size(), f) != raw.size()) die("short read");
    fclose(f);
    std::vector<float> out((size_t) rows * cols);
    const size_t rb = ggml_row_size(ty, cols);
    for (int64_t r = 0; r < rows; r++) ggml_get_type_traits(ty)->to_float(raw.data() + r * rb, out.data() + r * cols, cols);
    gguf_free(gg);
    ggml_free(gc);
    return out;
}

// C = A W^T in double, threaded over rows of A.
std::vector<double> ref_gemm(const std::vector<float> & A, const std::vector<float> & W, int64_t M, int64_t K, int64_t N) {
    std::vector<double> C((size_t) (M * N));
    const int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++) {
        th.emplace_back([&, t] {
            for (int64_t i = t; i < M; i += nt)
                for (int64_t j = 0; j < N; j++) {
                    const float * a = &A[i * K], * w = &W[j * K];
                    double s = 0;
                    for (int64_t k = 0; k < K; k++) s += (double) a[k] * w[k];
                    C[i * N + j] = s;
                }
        });
    }
    for (auto & x : th) x.join();
    return C;
}

double rel_frob(const std::vector<float> & got, const std::vector<double> & ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = got[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / den);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model, root = "kernels/bfp16_gemm/build/whole_array_bfp_acc_f32out";
    int64_t M = 512;
    int iters = 30;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--root"  && i + 1 < argc) root  = argv[++i];
        else if (a == "--m"     && i + 1 < argc) M     = atoll(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (model.empty())                  model = a;
        else die("unexpected argument " + a);
    }
    if (model.empty()) die("usage: test-npu-bfp16 <qwen3-q4.gguf> [--root <builds>] [--m 512|1024] [--iters n]");

    // layer 0's projections, fused the way the prefill will run them
    struct proj { const char * name; std::vector<const char *> parts; int64_t K = 0, N = 0; std::vector<float> W; };
    std::vector<proj> P = {
        { "q/k/v",   { "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight" } },
        { "output",  { "blk.0.attn_output.weight" } },
        { "gate/up", { "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight" } },
        { "down",    { "blk.0.ffn_down.weight" } },
    };
    for (proj & p : P) {
        for (const char * part : p.parts) {
            int64_t rows, cols;
            std::vector<float> w = load_tensor(model, part, rows, cols);
            if (p.K && p.K != cols) die("fused parts disagree on K");
            p.K = cols;
            p.N += rows;
            p.W.insert(p.W.end(), w.begin(), w.end());
        }
    }

    bfp16_tiling tile;  // 128 x 64 x 64, 8 columns: the float32-output build
    const int nth = std::max(1u, std::thread::hardware_concurrency());
    int failures = 0;

    // one context per K until K is a run-time parameter
    struct ctx_for_k { int64_t K; npu_bfp16 npu; };
    std::vector<std::unique_ptr<ctx_for_k>> ctxs;
    auto npu_for = [&](int64_t K, int64_t N) -> npu_bfp16 & {
        for (auto & c : ctxs) if (c->K == K) return c->npu;
        ctxs.push_back(std::make_unique<ctx_for_k>());
        ctxs.back()->K = K;
        const std::string dir = root + "/" + std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N) + "_128x64x64_c8";
        std::string err;
        if (!ctxs.back()->npu.open(dir + "/final.xclbin", err)) die(err);
        return ctxs.back()->npu;
    };

    std::mt19937 eng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 63);

    printf("M = %lld tokens, layer 0 of %s\n\n", (long long) M, model.c_str());
    printf("%-8s %-17s %-10s %-10s %-9s %-9s\n", "proj", "shape", "exact", "bfp16", "ms", "TFLOP/s");
    for (proj & p : P) {
        npu_bfp16 & npu = npu_for(p.K, p.N);
        std::string err;
        const std::string dir = root + "/" + std::to_string(M) + "x" + std::to_string(p.K) + "x" + std::to_string(p.N) + "_128x64x64_c8";
        const int si = npu.add_shape(dir + "/insts.bin", M, p.K, p.N, err);
        if (si < 0) die(err);

        std::vector<uint8_t> wpk;
        bfp16_pack_b(p.W.data(), p.N, p.K, tile, wpk, nth);
        const int wi = npu.add_weights(wpk, err);
        if (wi < 0) die(err);

        // activations: unit normal, with one value in 64 blown up 30x (outlier channels)
        std::vector<float> A((size_t) (M * p.K));
        for (float & v : A) v = nd(eng) * (pick(eng) == 0 ? 30.0f : 1.0f);
        std::vector<uint8_t> apk;
        bfp16_pack_a(A.data(), M, p.K, tile, apk, nth);
        if (!npu.set_a(si, apk, err)) die(err);

        std::vector<float> C((size_t) (M * p.N));
        std::vector<double> times;
        for (int it = 0; it < 10 + iters; it++) {
            double ms;
            if (!npu.run(si, wi, C.data(), err, &ms)) die(err);
            if (it >= 10) times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        const double med = times[times.size() / 2];

        // what the NPU was given, decoded, and the unrounded operands
        std::vector<float> Ad((size_t) (M * p.K)), Wd((size_t) (p.N * p.K));
        bfp16_unpack_a(apk.data(), M, p.K, tile, Ad.data());
        bfp16_unpack_b(wpk.data(), p.N, p.K, tile, Wd.data());
        const double e_exact = rel_frob(C, ref_gemm(Ad, Wd, M, p.K, p.N));
        const double e_fmt   = rel_frob(C, ref_gemm(A, p.W, M, p.K, p.N));

        char shape[32];
        snprintf(shape, sizeof(shape), "%lldx%lldx%lld", (long long) M, (long long) p.K, (long long) p.N);
        printf("%-8s %-17s %.2e   %.2e   %7.3f   %6.2f\n", p.name, shape, e_exact, e_fmt, med,
               2.0 * M * p.K * p.N / (med * 1e9));
        if (e_exact > 1e-5) { printf("FAIL %s: NPU result differs from the exact multiply of its operands\n", p.name); failures++; }
    }
    printf("\n%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
