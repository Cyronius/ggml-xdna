// Many GEMM shapes under ONE xclbin / hw context (whole_array_bfp_rtp).
//
//   mixk.exe <final.xclbin> <iters> <dir:M:K:N> [<dir:M:K:N> ...]
//
// Each shape uses its own build dir's insts.bin with the shared xclbin; the
// K-chunk count reaches the cores as a runtime parameter written by that
// insts.bin. Inputs are random bfp16 (fixed seed per shape), 128x64x64 tiles,
// linear layout; outputs are fp32 and every one is checked against a double
// CPU reference (frob_rel < 1e-5).
//
// Phases: (1) each shape alone, 3 dispatches, then 5 interleaved rounds, all
// verified; (2) all shapes interleaved round-robin for <iters> rounds, timed per
// dispatch; (3) each shape back-to-back for <iters> dispatches, timed; then 5
// more verified interleaved rounds. Timed dispatches do no host work in between.

#define BENCH_NO_MAIN
#include "bench_bfp16.cpp"

struct Shape {
    std::string dir;
    int M, K, N;
    xrt::bo bo_i, bo_a, bo_b, bo_c;
    xrt::run run;
    std::vector<double> ref;
    std::vector<double> t_mixed, t_alone;
    int bad = 0, runs = 0;
    bool tiled = false;  // a --c-tiled build (dir name has "_ct")
};

// --c-tiled C back to row-major: column c's region is M*N/8 floats; in it the
// 512 x 64 blocks go row block by row block, then N tiles c, c+8, ...
static std::vector<float> untile_c(const std::vector<float> & t, int M, int N) {
    const size_t J = N / 512, blk = 512 * 64;
    std::vector<float> C(t.size());
    for (int r = 0; r < M; r++)
        for (int s = 0; s < N; s++) {
            const size_t c = (s / 64) % 8, j = s / 512;
            C[(size_t) r * N + s] = t[c * (M * (size_t) N / 8) + ((r / 512) * J + j) * blk + (r % 512) * 64 + s % 64];
        }
    return C;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <xclbin> <iters> <dir:M:K:N>...\n", argv[0]); return 2; }
    const std::string xclbin = argv[1];
    const int iters = atoi(argv[2]);
    const int m = 128, k = 64, n = 64;

    xrt::device dev(0);
    xrt::xclbin xb(xclbin);
    dev.register_xclbin(xb);
    xrt::hw_context ctx(dev, xb.get_uuid());
    std::string kname;
    for (auto & kk : xb.get_kernels())
        if (kk.get_name().rfind("MLIR_AIE", 0) == 0) kname = kk.get_name();
    xrt::kernel kern(ctx, kname);

    std::vector<Shape> shapes;
    for (int i = 3; i < argc; i++) {
        std::string s = argv[i];
        Shape sh;
        size_t p1 = s.rfind(':'), p2 = s.rfind(':', p1 - 1), p3 = s.rfind(':', p2 - 1);
        sh.dir = s.substr(0, p3);
        sh.M = atoi(s.substr(p3 + 1).c_str());
        sh.K = atoi(s.substr(p2 + 1).c_str());
        sh.N = atoi(s.substr(p1 + 1).c_str());
        sh.tiled = sh.dir.find("_ct") != std::string::npos;
        shapes.push_back(sh);
    }
    for (size_t si = 0; si < shapes.size(); si++) {
        Shape & c = shapes[si];
        const int M = c.M, K = c.K, N = c.N;
        std::mt19937 rng(100 + (unsigned) si);
        std::normal_distribution<float> da(0.f, 1.f), dw(0.f, 0.05f);
        std::vector<float> A((size_t) M * K), Bt((size_t) N * K);
        for (auto & x : A) x = bf2f(f2bf(da(rng)));
        for (auto & x : Bt) x = dw(rng);
        auto Abfp = floatToBfp16Rne(M * K, A.data());
        auto Bbfp = floatToBfp16(8, N * K, Bt.data(), 0);
        auto Aq = bfp16ebs8ToFloat((int) Abfp.size(), Abfp.data());
        auto Bq = bfp16ebs8ToFloat((int) Bbfp.size(), Bbfp.data());
        auto Ash = linA(M, K, m, k, Abfp), Bsh = linB(N, K, n, k, 8, Bbfp);
        // A "_ctkv" build reads B from a bigger buffer (whole_array_bfp_rtp.py
        // --b-groups 4096 for the scores' K = 384, --b-kfull 4096 otherwise):
        // lay it out that way, the rest zeros.
        if (c.dir.find("_ctkv") != std::string::npos) {
            const size_t rowb = (size_t) K / 8 * 9;
            if (K == 384) {  // 512-row groups one after another, then zero groups up to 4096 rows
                Bsh.clear();
                for (int g = 0; g < N / 512; g++) {
                    std::vector<uint8_t> part(Bbfp.begin() + g * 512 * rowb, Bbfp.begin() + (g + 1) * 512 * rowb);
                    auto t = linB(512, K, n, k, 8, part);
                    Bsh.insert(Bsh.end(), t.begin(), t.end());
                }
                Bsh.resize((size_t) 4096 * rowb, 0);
            } else {         // each row padded to 4096 columns with zero blocks
                std::vector<uint8_t> wide((size_t) N * 4096 / 8 * 9, 0);
                for (int r = 0; r < N; r++) memcpy(wide.data() + (size_t) r * 4096 / 8 * 9, Bbfp.data() + r * rowb, rowb);
                Bsh = linB(N, 4096, n, k, 8, wide);
            }
        }
        c.ref = ref_gemm(M, K, N, Aq, Bq);
        auto insts = read_insts(c.dir + "/insts.bin");
        c.bo_i = xrt::bo(dev, insts.size() * 4, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
        c.bo_a = xrt::bo(dev, Ash.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
        c.bo_b = xrt::bo(dev, Bsh.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
        c.bo_c = xrt::bo(dev, (size_t) M * N * 4, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
        memcpy(c.bo_i.map<void *>(), insts.data(), insts.size() * 4);
        memcpy(c.bo_a.map<void *>(), Ash.data(), Ash.size());
        memcpy(c.bo_b.map<void *>(), Bsh.data(), Bsh.size());
        for (auto * bo : { &c.bo_i, &c.bo_a, &c.bo_b }) bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        c.run = xrt::run(kern);
        c.run.set_arg(0, 3u);
        c.run.set_arg(1, c.bo_i);
        c.run.set_arg(2, (unsigned) insts.size());
        c.run.set_arg(3, c.bo_a);
        c.run.set_arg(4, c.bo_b);
        c.run.set_arg(5, c.bo_c);
    }

    auto dispatch = [&](Shape & c, std::vector<double> * t, bool check) {
        const size_t cb = (size_t) c.M * c.N * 4;
        if (check) {
            memset(c.bo_c.map<void *>(), 0xFF, cb);  // NaN, so a skipped tile can't pass
            c.bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        auto t0 = clk::now();
        c.run.start();
        auto st = c.run.wait(std::chrono::milliseconds(20000));
        auto t1 = clk::now();
        if (st != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "%s: state %d\n", c.dir.c_str(), (int) st); exit(1); }
        if (t) t->push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (!check) return;
        c.bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::vector<float> C((size_t) c.M * c.N);
        memcpy(C.data(), c.bo_c.map<void *>(), cb);
        if (c.tiled) C = untile_c(C, c.M, c.N);
        Err e = compare(C, c.ref);
        c.runs++;
        if (!(e.frob_rel < 1e-5)) c.bad++;
    };

    // Verified passes (outputs reset to NaN, read back and checked):
    for (auto & c : shapes) for (int i = 0; i < 3; i++) dispatch(c, nullptr, true);      // (1)
    for (int r = 0; r < 5; r++) for (auto & c : shapes) dispatch(c, nullptr, true);      // interleaved
    // Timed passes, no host work between dispatches:
    for (int r = 0; r < 3; r++) for (auto & c : shapes) dispatch(c, nullptr, false);     // warm the mix
    for (int r = 0; r < iters; r++) for (auto & c : shapes) dispatch(c, &c.t_mixed, false);  // (2)
    for (auto & c : shapes) {                                                              // (3)
        for (int i = 0; i < 5; i++) dispatch(c, nullptr, false);
        for (int i = 0; i < iters; i++) dispatch(c, &c.t_alone, false);
    }
    // ...and verified again afterwards, interleaved, so a desync during the
    // timed passes would show up here.
    for (int r = 0; r < 5; r++) for (auto & c : shapes) dispatch(c, nullptr, true);

    int bad = 0;
    printf("one xclbin (%s), one hw context, %zu shapes, interleaved order as listed\n", xclbin.c_str(), shapes.size());
    printf("%-16s %5s %5s  %12s %12s  %s\n", "shape", "runs", "fail", "interleaved", "back-to-back", "TFLOP/s (b2b)");
    for (auto & c : shapes) {
        double mi = median(c.t_mixed), ma = median(c.t_alone);
        printf("%5dx%5dx%5d %5d %5d  %9.4f ms %9.4f ms  %.2f\n", c.M, c.K, c.N, c.runs, c.bad, mi, ma,
               2.0 * c.M * c.K * c.N / ma / 1e9);
        bad += c.bad;
    }
    printf("%s\n", bad ? "FAIL" : "all outputs match the reference");
    return bad ? 1 : 0;
}
