// Time and verify mlir-aie's whole-array bfp16ebs8 GEMM designs on the NPU.
//
//   bench_bfp16.exe <mixed|bfp|bfpacc> <build-dir> M K N m k n [key=value ...]
//     warmup=10 iters=50 runlist=8   timing
//     arnd=rne|trunc                 host rounding of A to bfp16 (bfp design; default rne)
//     data=gauss|ident               ident: B^T = identity (needs N == K), so C is
//                                    the core's own conversion of A (mixed design)
//     verify=1|0 prio=normal|high    skip the CPU reference; raise process priority
//     xclbin=<path>                  load this xclbin instead of <build-dir>/final.xclbin
//                                    (insts.bin still comes from <build-dir>)
//     layout=stock|paired|lin        paired: A and B tiles pair-interleaved for
//                                    mm_bfp.cc's PAIRED kernel (bfp design);
//                                    lin: paired, and every tile contiguous in
//                                    consumption order (whole_array_bfp_lin)
//
// mixed: A bf16 (M x K row-major), B bfp16ebs8 (N x K, i.e. B^T, shuffled), C bf16.
// bfp:   A, B, C all bfp16ebs8, shuffled on the host (helper.h).
// bfpacc: A, B as bfp; C bf16 row-major (whole_array_bfp_acc, fp32/bf16 partial sums).
// bfpacc32: same, C fp32 row-major (whole_array_bfp_acc --acc 33).
//
// Timing: one xrt::run reused, start()+wait() per dispatch, the first `warmup`
// dispatches discarded, then the median of `iters`. A second pass puts
// `runlist` runs in one xrt::runlist and reports total/runlist, which takes
// most of the ~0.1 ms per-submission cost out of the figure.
//
// Correctness: the output of the first timed dispatch is compared with a CPU
// reference (double accumulation) over exactly the bfp16 operands the NPU was
// given; every later dispatch's output must be bit-identical to it.

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_kernel.h"

#include "helper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>
#define NOMINMAX
#include <windows.h>

using clk = std::chrono::steady_clock;

static uint16_t f2bf(float f) {  // round to nearest even
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}
static float bf2f(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// Same byte layout as helper.h's floatToBfp16 (exponent byte, then 8 int8
// mantissas, value = q * 2^(e-127) / 64), but round-to-nearest-even instead of
// truncation. The shared exponent is the block's largest float exponent; a
// mantissa that rounds up to 128 saturates at 127.
static std::vector<uint8_t> floatToBfp16Rne(int size, const float * x) {
    std::vector<uint8_t> r((size_t) size / 8 * 9);
    for (int b = 0; b < size / 8; b++) {
        unsigned maxe = 0;
        for (int i = 0; i < 8; i++) {
            uint32_t u;
            memcpy(&u, &x[b * 8 + i], 4);
            maxe = std::max(maxe, (u >> 23) & 0xFF);
        }
        r[(size_t) b * 9] = (uint8_t) maxe;
        for (int i = 0; i < 8; i++) {
            double q = std::nearbyint(std::ldexp((double) x[b * 8 + i], 133 - (int) maxe));
            q = std::min(127.0, std::max(-128.0, q));
            r[(size_t) b * 9 + 1 + i] = (uint8_t) (int8_t) q;
        }
    }
    return r;
}

// Like helper.h's shuffleMatrixForBfp16ebs8, but within each tile the 8x8
// subtiles go (row pair p, subtile column sx, row 2p then 2p+1), so the PAIRED
// kernel reads (z,i),(z+1,i) from one stream. W/H in elements, tile tw x th.
static std::vector<uint8_t> shufflePaired(size_t W, size_t H, size_t tw, size_t th,
                                          const std::vector<uint8_t> & bfp) {
    const size_t Wb = W / 8 * 9, twb = tw / 8 * 9;
    std::vector<uint8_t> res(Wb * H);
    for (size_t ty = 0; ty < H; ty += th)
        for (size_t tx = 0; tx < Wb; tx += twb) {
            size_t cnt = 0;
            for (size_t sp = 0; sp < th; sp += 16)
                for (size_t sx = 0; sx < twb; sx += 9)
                    for (size_t half = 0; half < 2; half++)
                        for (size_t i = 0; i < 8; i++)
                            for (size_t j = 0; j < 9; j++, cnt++)
                                res[(ty + cnt / twb) * Wb + tx + cnt % twb] =
                                    bfp[(ty + sp + half * 8 + i) * Wb + tx + sx + j];
        }
    return res;
}

// One tile of a row-major bfp16 matrix (Wb bytes per row), in the order the
// core reads it: 8x8 subtiles, row-major (stock) or pair-interleaved.
static void emit_tile(const std::vector<uint8_t> & bfp, size_t Wb, size_t r0, size_t c0b, size_t th,
                      size_t twb, bool paired, std::vector<uint8_t> & out) {
    if (paired) {
        for (size_t sp = 0; sp < th; sp += 16)
            for (size_t sx = 0; sx < twb; sx += 9)
                for (size_t half = 0; half < 2; half++)
                    for (size_t i = 0; i < 8; i++)
                        for (size_t j = 0; j < 9; j++) out.push_back(bfp[(r0 + sp + half * 8 + i) * Wb + c0b + sx + j]);
    } else {
        for (size_t sy = 0; sy < th; sy += 8)
            for (size_t sx = 0; sx < twb; sx += 9)
                for (size_t i = 0; i < 8; i++)
                    for (size_t j = 0; j < 9; j++) out.push_back(bfp[(r0 + sy + i) * Wb + c0b + sx + j]);
    }
}

// whole_array_bfp_lin's host layouts (see its _lin_tap comment).
static std::vector<uint8_t> linA(int M, int K, int m, int k, const std::vector<uint8_t> & bfp) {
    std::vector<uint8_t> out;
    out.reserve(bfp.size());
    for (int rb = 0; rb < M / m; rb++)
        for (int kk = 0; kk < K / k; kk++) emit_tile(bfp, K / 8 * 9, (size_t) rb * m, (size_t) kk * k / 8 * 9, m, k / 8 * 9, true, out);
    return out;
}
static std::vector<uint8_t> linB(int N, int K, int n, int k, int cols, const std::vector<uint8_t> & bfp) {
    std::vector<uint8_t> out;
    out.reserve(bfp.size());
    for (int c = 0; c < cols; c++)
        for (int t = 0; t < N / n / cols; t++)
            for (int kk = 0; kk < K / k; kk++)
                emit_tile(bfp, K / 8 * 9, (size_t) (c + cols * t) * n, (size_t) kk * k / 8 * 9, n, k / 8 * 9, true, out);
    return out;
}

static std::vector<uint32_t> read_insts(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<uint32_t> w(b.size() / 4);
    memcpy(w.data(), b.data(), w.size() * 4);
    return w;
}

// C[M x N] = A[M x K] * Bt[N x K]^T, double accumulation, threaded over rows.
static std::vector<double> ref_gemm(int M, int K, int N, const std::vector<float> & A,
                                    const std::vector<float> & Bt) {
    std::vector<double> C((size_t) M * N);
    unsigned nt = std::max(1u, std::thread::hardware_concurrency() / 2);
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; t++) {
        th.emplace_back([&, t] {
            for (int i = (int) t; i < M; i += (int) nt) {
                const float * a = &A[(size_t) i * K];
                for (int j = 0; j < N; j++) {
                    const float * b = &Bt[(size_t) j * K];
                    double s = 0;
                    for (int kk = 0; kk < K; kk++) s += (double) a[kk] * b[kk];
                    C[(size_t) i * N + j] = s;
                }
            }
        });
    }
    for (auto & x : th) x.join();
    return C;
}

struct Err { double frob_rel, max_abs_over_max_ref, max_abs, cos, slope; };
static Err compare(const std::vector<float> & got, const std::vector<double> & ref) {
    double num = 0, den = 0, maxabs = 0, maxref = 0, dot = 0, gg = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        double d = (double) got[i] - ref[i];
        num += d * d; den += ref[i] * ref[i];
        maxabs = std::max(maxabs, std::fabs(d));
        maxref = std::max(maxref, std::fabs(ref[i]));
        dot += got[i] * ref[i]; gg += (double) got[i] * got[i];
    }
    return { std::sqrt(num / den), maxabs / maxref, maxabs, dot / std::sqrt(gg * den), dot / den };
}
static void print_err(const char * what, const Err & e) {
    printf("vs %-26s frob_rel %.3e  max|err|/max|ref| %.3e  cos %.7f  slope %.4f\n", what, e.frob_rel,
           e.max_abs_over_max_ref, e.cos, e.slope);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

#ifndef BENCH_NO_MAIN  // ctxswitch.cpp reuses the helpers above
int main(int argc, char ** argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: %s <mixed|bfp|bfpacc|bfpacc32> <build-dir> M K N m k n [warmup=] [iters=] [runlist=] [arnd=] [data=]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1], dir = argv[2];
    const int M = atoi(argv[3]), K = atoi(argv[4]), N = atoi(argv[5]);
    const int m = atoi(argv[6]), k = atoi(argv[7]), n = atoi(argv[8]);
    std::map<std::string, std::string> opt = { { "warmup", "10" }, { "iters", "50" }, { "runlist", "8" },
                                               { "arnd", "rne" }, { "data", "gauss" }, { "layout", "stock" }, { "verify", "1" }, { "prio", "normal" }, { "xclbin", "" } };
    for (int i = 9; i < argc; i++) {
        std::string a = argv[i];
        size_t eq = a.find('=');
        if (eq == std::string::npos || !opt.count(a.substr(0, eq))) { fprintf(stderr, "bad option %s\n", argv[i]); return 2; }
        opt[a.substr(0, eq)] = a.substr(eq + 1);
    }
    const int warmup = atoi(opt["warmup"].c_str()), iters = atoi(opt["iters"].c_str());
    const int rl_n = atoi(opt["runlist"].c_str());
    const bool mixed = mode == "mixed", cbf16 = mixed || mode == "bfpacc", cf32 = mode == "bfpacc32", arne = opt["arnd"] == "rne", ident = opt["data"] == "ident";
    const bool lin = opt["layout"] == "lin", paired = lin || opt["layout"] == "paired";
    if (ident && N != K) { fprintf(stderr, "data=ident needs N == K\n"); return 2; }
    if (opt["prio"] == "high") SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // ---- data: activations ~ N(0,1), weights ~ N(0,0.05) ------------------
    std::mt19937 rng(1234);
    std::normal_distribution<float> da(0.f, 1.f), dw(0.f, 0.05f);
    std::vector<float> A((size_t) M * K), Bt((size_t) N * K);
    for (auto & x : A) x = bf2f(f2bf(da(rng)));   // activations are bf16 on the host
    for (size_t i = 0; i < Bt.size(); i++) Bt[i] = ident ? (i / K == i % K ? 1.f : 0.f) : dw(rng);

    // B^T to bfp16 (helper.h; weights are converted once, rounding mode is a
    // load-time choice), dequantised copy for the reference, then shuffled.
    std::vector<uint8_t> Bbfp = floatToBfp16(8, N * K, Bt.data(), 0);
    std::vector<float> Bq = bfp16ebs8ToFloat((int) Bbfp.size(), Bbfp.data());
    std::vector<uint8_t> Bsh = lin ? linB(N, K, n, k, 8, Bbfp)
                               : paired ? shufflePaired(K, N, k, n, Bbfp) : shuffleMatrixForBfp16ebs8(K, N, k, n, Bbfp);

    std::vector<float> Aq;               // bfp mode: A as the NPU sees it
    std::vector<uint8_t> Ash;            // bfp mode: A shuffled bfp16
    std::vector<uint16_t> Abf;           // mixed mode: A bf16
    if (mixed) {
        Abf.resize(A.size());
        for (size_t i = 0; i < A.size(); i++) Abf[i] = f2bf(A[i]);
    } else {
        std::vector<uint8_t> Abfp = arne ? floatToBfp16Rne(M * K, A.data()) : floatToBfp16(8, M * K, A.data(), 0);
        Aq = bfp16ebs8ToFloat((int) Abfp.size(), Abfp.data());
        Ash = lin ? linA(M, K, m, k, Abfp) : paired ? shufflePaired(K, M, k, m, Abfp) : shuffleMatrixForBfp16ebs8(K, M, k, m, Abfp);
    }

    const size_t a_bytes = mixed ? A.size() * 2 : Ash.size();
    const size_t b_bytes = Bsh.size();
    const size_t c_bytes = cf32 ? (size_t) M * N * 4 : cbf16 ? (size_t) M * N * 2 : (size_t) M * N * 9 / 8;

    // ---- XRT -------------------------------------------------------------
    auto insts = read_insts(dir + "/insts.bin");
    xrt::device dev(0);
    xrt::xclbin xb(opt["xclbin"].empty() ? dir + "/final.xclbin" : opt["xclbin"]);
    dev.register_xclbin(xb);
    xrt::hw_context ctx(dev, xb.get_uuid());
    std::string kname;
    for (auto & kk : xb.get_kernels())
        if (kk.get_name().rfind("MLIR_AIE", 0) == 0) kname = kk.get_name();
    xrt::kernel kern(ctx, kname);

    xrt::bo bo_i(dev, insts.size() * 4, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
    xrt::bo bo_a(dev, a_bytes, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
    xrt::bo bo_b(dev, b_bytes, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
    xrt::bo bo_c(dev, c_bytes, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
    memcpy(bo_i.map<void *>(), insts.data(), insts.size() * 4);
    memcpy(bo_a.map<void *>(), mixed ? (const void *) Abf.data() : (const void *) Ash.data(), a_bytes);
    memcpy(bo_b.map<void *>(), Bsh.data(), b_bytes);
    memset(bo_c.map<void *>(), 0, c_bytes);
    bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto make_run = [&] {
        xrt::run r(kern);
        r.set_arg(0, 3u);
        r.set_arg(1, bo_i);
        r.set_arg(2, (unsigned) insts.size());
        r.set_arg(3, bo_a);
        r.set_arg(4, bo_b);
        r.set_arg(5, bo_c);
        return r;
    };

    // ---- single-dispatch timing ------------------------------------------
    std::vector<uint8_t> first(c_bytes);
    int mismatched_runs = 0;
    std::vector<double> ms;
    xrt::run run = make_run();
    for (int it = 0; it < warmup + iters; it++) {
        auto t0 = clk::now();
        run.start();
        auto st = run.wait(std::chrono::milliseconds(20000));
        auto t1 = clk::now();
        if (st != ERT_CMD_STATE_COMPLETED) {
            fprintf(stderr, "run %d: state %d\n", it, (int) st);
            return 1;
        }
        if (it < warmup) continue;
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (it == warmup) memcpy(first.data(), bo_c.map<void *>(), c_bytes);
        else if (memcmp(first.data(), bo_c.map<void *>(), c_bytes) != 0) mismatched_runs++;
    }

    // ---- runlist timing ----------------------------------------------------
    std::vector<double> rl_ms;
    if (rl_n > 1) {
        std::vector<xrt::run> runs;
        for (int i = 0; i < rl_n; i++) runs.push_back(make_run());
        for (int it = 0; it < 3 + std::max(10, iters / 2); it++) {
            xrt::runlist rl(ctx);
            for (auto & r : runs) rl.add(r);
            auto t0 = clk::now();
            rl.execute();
            rl.wait(std::chrono::milliseconds(60000));
            auto t1 = clk::now();
            if (it >= 3) rl_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / rl_n);
        }
    }

    const double flop = 2.0 * M * K * N;
    const double med = median(ms);
    std::vector<double> s = ms; std::sort(s.begin(), s.end());
    const double mn = s.front(), p10 = s[(size_t) (0.1 * (s.size() - 1))], p90 = s[(size_t) (0.9 * (s.size() - 1))];
    printf("design=%s M=%d K=%d N=%d tile=%dx%dx%d insts=%zu\n", mode.c_str(), M, K, N, m, k, n, insts.size());
    printf("single: median %.4f ms  min %.4f  p10 %.4f  p90 %.4f  (n=%zu, warmup %d)  -> %.2f TFLOP/s (min %.2f)\n",
           med, mn, p10, p90, ms.size(), warmup, flop / med / 1e9, flop / mn / 1e9);
    double rmed = 0;
    if (!rl_ms.empty()) {
        rmed = median(rl_ms);
        printf("runlist x%d: median %.4f ms/run (n=%zu)  -> %.2f TFLOP/s\n", rl_n, rmed, rl_ms.size(), flop / rmed / 1e9);
    }
    printf("repeat determinism: %d of %zu timed outputs differ from the first\n", mismatched_runs, ms.size() - 1);

    // ---- correctness ---------------------------------------------------------
    std::vector<float> C((size_t) M * N);
    if (cf32) {
        memcpy(C.data(), first.data(), C.size() * 4);
    } else if (cbf16) {
        const uint16_t * c16 = (const uint16_t *) first.data();
        for (size_t i = 0; i < C.size(); i++) C[i] = bf2f(c16[i]);
    } else {
        std::vector<uint8_t> un = shuffleMatrixForBfp16ebs8(N, M, n, m, first, true);
        C = bfp16ebs8ToFloat((int) un.size(), un.data());
    }
    if (opt["verify"] == "0") {
        printf("RESULT %s %dx%dx%d tile %dx%dx%d median_ms %.4f tflops %.2f rl_ms %.4f rl_tflops %.2f p10_ms %.4f noverify\n",
               mode.c_str(), M, K, N, m, k, n, med, flop / med / 1e9, rmed, rmed > 0 ? flop / rmed / 1e9 : 0.0, p10);
        return mismatched_runs == 0 ? 0 : 1;
    }
    size_t nonfinite = 0;
    for (float x : C) if (!std::isfinite(x)) nonfinite++;
    if (nonfinite) printf("NONFINITE outputs: %zu\n", nonfinite);
    auto t0 = clk::now();
    Err e;
    if (!mixed) {
        auto R = ref_gemm(M, K, N, Aq, Bq);
        e = compare(C, R);
        print_err(arne ? "ref(A_bfp16rne x B_bfp16)" : "ref(A_bfp16trn x B_bfp16)", e);
        if (cbf16) {
            // floor: the reference itself rounded once to bf16 (RNE)
            std::vector<float> Rb(R.size());
            for (size_t i = 0; i < R.size(); i++) Rb[i] = bf2f(f2bf((float) R[i]));
            print_err("bf16(ref) [output-rounding floor]", compare(Rb, R));
        }
    } else {
        std::vector<uint8_t> At = floatToBfp16(8, M * K, A.data(), 0), Ar = floatToBfp16Rne(M * K, A.data());
        std::vector<float> Atq = bfp16ebs8ToFloat((int) At.size(), At.data());
        std::vector<float> Arq = bfp16ebs8ToFloat((int) Ar.size(), Ar.data());
        if (ident) {
            // C should be the core's bfp16 conversion of A, exactly (bf16 holds it).
            size_t eq_t = 0, eq_r = 0;
            for (size_t i = 0; i < C.size(); i++) { eq_t += C[i] == Atq[i]; eq_r += C[i] == Arq[i]; }
            printf("identity probe: C == trunc(A) for %.4f%%, C == rne(A) for %.4f%% of %zu elements\n",
                   100.0 * eq_t / C.size(), 100.0 * eq_r / C.size(), C.size());
            for (size_t i = 0, shown = 0; i < C.size() && shown < 6; i++) {
                if (C[i] != Arq[i] && C[i] != Atq[i]) {
                    size_t b0 = i / 8 * 8;
                    printf("  mismatch @%zu: A=%.6g core=%.6g rne=%.6g trunc=%.6g  block:", i, A[i], C[i], Arq[i], Atq[i]);
                    for (size_t j = b0; j < b0 + 8; j++) printf(" %.4g", A[j]);
                    printf("\n");
                    shown++;
                }
            }
        }
        e = compare(C, ref_gemm(M, K, N, A, Bq));
        print_err("ref(A_bf16 x B_bfp16)", e);
        print_err("ref(A_bfp16trn x B_bfp16)", compare(C, ref_gemm(M, K, N, Atq, Bq)));
        print_err("ref(A_bfp16rne x B_bfp16)", compare(C, ref_gemm(M, K, N, Arq, Bq)));
    }
    printf("verify took %.1f s\n", std::chrono::duration<double>(clk::now() - t0).count());
    printf("RESULT %s %dx%dx%d tile %dx%dx%d median_ms %.4f tflops %.2f rl_ms %.4f rl_tflops %.2f frob %.3e maxrel %.3e slope %.4f\n",
           mode.c_str(), M, K, N, m, k, n, med, flop / med / 1e9, rmed, rmed > 0 ? flop / rmed / 1e9 : 0.0,
           e.frob_rel, e.max_abs_over_max_ref, e.slope);
    return (nonfinite == 0 && mismatched_runs == 0) ? 0 : 1;
}
#endif  // BENCH_NO_MAIN
