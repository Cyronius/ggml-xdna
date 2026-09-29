// whole_array_bfp_rtp's output modes on the NPU, one shape per run (step 2 of
// .claude/plans/smaller-npu-outputs.md):
//
//   modes.exe <xclbin> <iters> <M> <K> <N> <mode> <mode 0 build dir> <mode build dir>
//
// The same random operands go through both builds under one xclbin (--c-tiled
// builds). Mode 0's fp32 output must match a double reference, as in mixk.
// The mode's bf16 output, read back with hybrid/bfp16_pack's bfp16_c16_row
// (the reader the prefill uses), must give:
//   mode 1: mode 0's output rounded to bf16, bit for bit
//   mode 2: SiLU(gate) * up from mode 0's output, taking column 16i + j as
//           gate and 16i + 8 + j as up (bfp16_interleave_gate_up's order),
//           within the kernel's exp error plus one bf16 rounding
// Then the two builds are timed, alternating, and the medians printed.

#define BENCH_NO_MAIN
#include "bench_bfp16.cpp"

#include "bfp16_pack.h"

struct Build {
    xrt::bo bo_i, bo_c;
    xrt::run run;
};

int main(int argc, char ** argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: %s <xclbin> <iters> <M> <K> <N> <mode> <mode 0 dir> <mode dir>\n", argv[0]);
        return 2;
    }
    const std::string xclbin = argv[1];
    const int iters = atoi(argv[2]), M = atoi(argv[3]), K = atoi(argv[4]), N = atoi(argv[5]), mode = atoi(argv[6]);
    const int m = 128, k = 64, n = 64;
    if (mode != 1 && mode != 2) { fprintf(stderr, "mode must be 1 or 2\n"); return 2; }

    xrt::device dev(0);
    xrt::xclbin xb(xclbin);
    dev.register_xclbin(xb);
    xrt::hw_context ctx(dev, xb.get_uuid());
    std::string kname;
    for (auto & kk : xb.get_kernels())
        if (kk.get_name().rfind("MLIR_AIE", 0) == 0) kname = kk.get_name();
    xrt::kernel kern(ctx, kname);

    std::mt19937 rng(7);
    std::normal_distribution<float> da(0.f, 1.f), dw(0.f, 0.05f);
    std::vector<float> A((size_t) M * K), Bt((size_t) N * K);
    for (auto & x : A) x = bf2f(f2bf(da(rng)));
    for (auto & x : Bt) x = dw(rng);
    auto Abfp = floatToBfp16Rne(M * K, A.data());
    auto Bbfp = floatToBfp16(8, N * K, Bt.data(), 0);
    auto Aq = bfp16ebs8ToFloat((int) Abfp.size(), Abfp.data());
    auto Bq = bfp16ebs8ToFloat((int) Bbfp.size(), Bbfp.data());
    auto Ash = linA(M, K, m, k, Abfp), Bsh = linB(N, K, n, k, 8, Bbfp);
    const std::vector<double> ref = ref_gemm(M, K, N, Aq, Bq);

    xrt::bo bo_a(dev, Ash.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
    xrt::bo bo_b(dev, Bsh.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
    memcpy(bo_a.map<void *>(), Ash.data(), Ash.size());
    memcpy(bo_b.map<void *>(), Bsh.data(), Bsh.size());
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const size_t cb = (size_t) M * N * 4;
    Build b[2];
    for (int i = 0; i < 2; i++) {
        auto insts = read_insts(std::string(argv[7 + i]) + "/insts.bin");
        b[i].bo_i = xrt::bo(dev, insts.size() * 4, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
        b[i].bo_c = xrt::bo(dev, cb, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
        memcpy(b[i].bo_i.map<void *>(), insts.data(), insts.size() * 4);
        b[i].bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        b[i].run = xrt::run(kern);
        b[i].run.set_arg(0, 3u);
        b[i].run.set_arg(1, b[i].bo_i);
        b[i].run.set_arg(2, (unsigned) insts.size());
        b[i].run.set_arg(3, bo_a);
        b[i].run.set_arg(4, bo_b);
        b[i].run.set_arg(5, b[i].bo_c);
    }
    auto dispatch = [&](Build & x, bool poison) {
        if (poison) {  // NaN everywhere, so a value the NPU never wrote can't pass
            memset(x.bo_c.map<void *>(), 0xFF, cb);
            x.bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        auto t0 = clk::now();
        x.run.start();
        auto st = x.run.wait(std::chrono::milliseconds(20000));
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        if (st != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "state %d\n", (int) st); exit(1); }
        if (poison) x.bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return ms;
    };

    bfp16_tiling t;
    t.m = m; t.k = k; t.n = n;
    const int n_out = mode == 2 ? N / 2 : N;
    int bad = 0;
    for (int round = 0; round < 3; round++) {
        dispatch(b[0], true);
        dispatch(b[1], true);
        std::vector<float> C0((size_t) M * N), got((size_t) M * n_out);
        const float * c0 = b[0].bo_c.map<const float *>(), * c1 = b[1].bo_c.map<const float *>();
        for (int r = 0; r < M; r++) {
            bfp16_c_row(c0, M, N, t, r, N, C0.data() + (size_t) r * N);
            bfp16_c16_row(c1, M, N, t, mode, r, got.data() + (size_t) r * n_out);
        }
        const Err e = compare(C0, ref);
        size_t wrong = 0, nonfinite = 0, exact = 0;
        double worst = 0;
        for (int r = 0; r < M; r++)
            for (int f = 0; f < n_out; f++) {
                const float g = got[(size_t) r * n_out + f];
                if (!std::isfinite(g)) { nonfinite++; continue; }
                if (mode == 1) {
                    const float want = bf2f(f2bf(C0[(size_t) r * N + f]));
                    if (memcmp(&g, &want, 4) != 0) wrong++;
                    else exact++;
                } else {
                    const int64_t gc = (f / 8) * 16 + f % 8;
                    const double x = C0[(size_t) r * N + gc], u = C0[(size_t) r * N + gc + 8];
                    const double want = x / (1.0 + std::exp(-x)) * u;
                    if (g == bf2f(f2bf((float) want))) exact++;
                    // half a bf16 step for the rounding, plus the exp's 8.9e-5
                    // and float arithmetic, relative; the absolute floor covers
                    // products near zero
                    const double err = std::fabs(g - want), tol = std::fabs(want) * (1.0 / 256 + 2e-4) + 1e-30;
                    worst = std::max(worst, err / (std::fabs(want) + 1e-30));
                    if (err > tol) wrong++;
                }
            }
        const bool ok = e.frob_rel < 1e-5 && wrong == 0 && nonfinite == 0;
        bad += !ok;
        printf("round %d: mode 0 frob_rel %.2e | mode %d: %zu wrong, %zu not finite, %zu of %zu exactly the reference's "
               "bf16%s\n",
               round, e.frob_rel, mode, wrong, nonfinite, exact, (size_t) M * n_out,
               mode == 2 ? (" (worst relative error " + std::to_string(worst) + ")").c_str() : "");
    }

    std::vector<double> t0, t1;
    for (int i = 0; i < 5; i++) { dispatch(b[0], false); dispatch(b[1], false); }
    for (int i = 0; i < iters; i++) {
        t0.push_back(dispatch(b[0], false));
        t1.push_back(dispatch(b[1], false));
    }
    // Under other load a dispatch only gets slower, so the fastest runs are
    // the steadiest comparison; the median is printed too.
    auto pct = [](std::vector<double> v, double q) {
        std::sort(v.begin(), v.end());
        return v[(size_t) (q * (v.size() - 1))];
    };
    printf("%dx%dx%d, %d runs each, alternating: mode 0 min %.4f / p10 %.4f / median %.4f ms, mode %d min %.4f / p10 "
           "%.4f / median %.4f ms: %+.1f%% at the min, %+.1f%% at p10, %+.1f%% at the median\n",
           M, K, N, iters, pct(t0, 0), pct(t0, 0.1), median(t0), mode, pct(t1, 0), pct(t1, 0.1), median(t1),
           100.0 * (pct(t1, 0) / pct(t0, 0) - 1), 100.0 * (pct(t1, 0.1) / pct(t0, 0.1) - 1),
           100.0 * (median(t1) / median(t0) - 1));
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
