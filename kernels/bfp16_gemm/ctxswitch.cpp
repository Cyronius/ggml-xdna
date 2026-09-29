// What does alternating between two hardware contexts cost?
//
//   ctxswitch.exe <dirA> MA KA NA <dirB> MB KB NB [iters=200] [warmup=20]
//
// Both dirs are whole_array_bfp_acc --acc 33 builds (128x64x64, fp32 out). Both
// xclbins are registered and both hw contexts created up front, and stay alive
// for the whole run. Per-dispatch wall time (start + wait) is measured for:
//   A only, B only (back-to-back on one context) and A,B,A,B... (alternating).
// Every output is checked against a double-precision CPU reference.

#define BENCH_NO_MAIN
#include "bench_bfp16.cpp"

struct Ctx {
    std::string dir;
    int M, K, N;
    xrt::hw_context hw;
    xrt::kernel kern;
    xrt::bo bo_i, bo_a, bo_b, bo_c;
    xrt::run run;
    std::vector<double> ref;
    size_t c_elems;
};

static void setup(xrt::device & dev, Ctx & c, unsigned seed) {
    const int m = 128, k = 64, n = 64, M = c.M, K = c.K, N = c.N;
    std::mt19937 rng(seed);
    std::normal_distribution<float> da(0.f, 1.f), dw(0.f, 0.05f);
    std::vector<float> A((size_t) M * K), Bt((size_t) N * K);
    for (auto & x : A) x = bf2f(f2bf(da(rng)));
    for (auto & x : Bt) x = dw(rng);
    auto Abfp = floatToBfp16Rne(M * K, A.data());
    auto Bbfp = floatToBfp16(8, N * K, Bt.data(), 0);
    auto Aq = bfp16ebs8ToFloat((int) Abfp.size(), Abfp.data());
    auto Bq = bfp16ebs8ToFloat((int) Bbfp.size(), Bbfp.data());
    const int cols = getenv("CTX_COLS") ? atoi(getenv("CTX_COLS")) : 8;  // array columns the builds use
    auto Ash = linA(M, K, m, k, Abfp), Bsh = linB(N, K, n, k, cols, Bbfp);
    c.ref = ref_gemm(M, K, N, Aq, Bq);
    c.c_elems = (size_t) M * N;

    auto insts = read_insts(c.dir + "/insts.bin");
    xrt::xclbin xb(c.dir + "/final.xclbin");
    dev.register_xclbin(xb);
    c.hw = xrt::hw_context(dev, xb.get_uuid());
    std::string kname;
    for (auto & kk : xb.get_kernels())
        if (kk.get_name().rfind("MLIR_AIE", 0) == 0) kname = kk.get_name();
    c.kern = xrt::kernel(c.hw, kname);
    c.bo_i = xrt::bo(dev, insts.size() * 4, XCL_BO_FLAGS_CACHEABLE, c.kern.group_id(1));
    c.bo_a = xrt::bo(dev, Ash.size(), XRT_BO_FLAGS_HOST_ONLY, c.kern.group_id(3));
    c.bo_b = xrt::bo(dev, Bsh.size(), XRT_BO_FLAGS_HOST_ONLY, c.kern.group_id(4));
    c.bo_c = xrt::bo(dev, c.c_elems * 4, XRT_BO_FLAGS_HOST_ONLY, c.kern.group_id(5));
    memcpy(c.bo_i.map<void *>(), insts.data(), insts.size() * 4);
    memcpy(c.bo_a.map<void *>(), Ash.data(), Ash.size());
    memcpy(c.bo_b.map<void *>(), Bsh.data(), Bsh.size());
    for (auto * bo : { &c.bo_i, &c.bo_a, &c.bo_b }) bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    c.run = xrt::run(c.kern);
    c.run.set_arg(0, 3u);
    c.run.set_arg(1, c.bo_i);
    c.run.set_arg(2, (unsigned) insts.size());
    c.run.set_arg(3, c.bo_a);
    c.run.set_arg(4, c.bo_b);
    c.run.set_arg(5, c.bo_c);
}

static double dispatch(Ctx & c) {
    memset(c.bo_c.map<void *>(), 0, c.c_elems * 4);  // so a skipped run can't pass
    c.bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t0 = clk::now();
    c.run.start();
    auto st = c.run.wait(std::chrono::milliseconds(20000));
    auto t1 = clk::now();
    if (st != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "%s: state %d\n", c.dir.c_str(), (int) st); exit(1); }
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static bool check(Ctx & c) {
    c.bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::vector<float> C(c.c_elems);
    memcpy(C.data(), c.bo_c.map<void *>(), c.c_elems * 4);
    return compare(C, c.ref).frob_rel < 1e-5;
}

int main(int argc, char ** argv) {
    if (argc < 9) { fprintf(stderr, "usage: %s dirA MA KA NA dirB MB KB NB [iters] [warmup]\n", argv[0]); return 2; }
    const int iters = argc > 9 ? atoi(argv[9]) : 200, warmup = argc > 10 ? atoi(argv[10]) : 20;
    xrt::device dev(0);
    Ctx a{ argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]) };
    Ctx b{ argv[5], atoi(argv[6]), atoi(argv[7]), atoi(argv[8]) };
    setup(dev, a, 1);
    setup(dev, b, 2);
    printf("both contexts created: A %dx%dx%d, B %dx%dx%d\n", a.M, a.K, a.N, b.M, b.K, b.N);

    int bad = 0;
    auto series = [&](const char * name, std::vector<Ctx *> order) {
        std::vector<double> t[2];
        for (int it = 0; it < warmup + iters; it++) {
            Ctx * c = order[it % order.size()];
            double ms = dispatch(*c);
            if (!check(*c)) bad++;
            if (it >= warmup) t[c == &b].push_back(ms);
        }
        printf("%-12s", name);
        for (int i = 0; i < 2; i++)
            if (!t[i].empty()) {
                auto s = t[i];
                std::sort(s.begin(), s.end());
                printf("  %s: median %.4f ms  p10 %.4f  p90 %.4f (n=%zu)", i ? "B" : "A", median(s),
                       s[s.size() / 10], s[s.size() * 9 / 10], s.size());
            }
        printf("\n");
    };
    series("A only", { &a });
    series("B only", { &b });
    series("alternating", { &a, &b });
    series("A only", { &a });
    printf("outputs failing the reference check: %d\n", bad);
    return bad ? 1 : 0;
}
