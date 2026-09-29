// Can the NPU compute straight out of memory it didn't allocate? (Step 0 of
// .claude/plans/backend-size-aware.md.)
//
//   userptr.exe <xclbin> <iters> <M> <K> <N> <build dir>
//
// Runs one whole_array_bfp_rtp build (output mode 0) twice on the same random
// operands: once with every buffer allocated by XRT as usual, once with the
// activations (A) and the output (C) in xrt::ext::bo buffers over ordinary
// page-aligned host memory the program allocated itself. The two outputs must
// match bit for bit, and the first must match a double reference as in mixk.
// Then both are timed, alternating. The output buffer is also read by the CPU
// straight from the user memory, to see whether it needs a sync.

#define BENCH_NO_MAIN
#include "bench_bfp16.cpp"

#include "bfp16_pack.h"
#include "xrt/experimental/xrt_ext.h"

#include <malloc.h>

int main(int argc, char ** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s <xclbin> <iters> <M> <K> <N> <build dir>\n", argv[0]);
        return 2;
    }
    const std::string xclbin = argv[1];
    const int iters = atoi(argv[2]), M = atoi(argv[3]), K = atoi(argv[4]), N = atoi(argv[5]);
    const int m = 128, k = 64, n = 64;

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
    const size_t cb = (size_t) M * N * 4;

    auto insts = read_insts(std::string(argv[6]) + "/insts.bin");
    xrt::bo bo_i(dev, insts.size() * 4, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
    memcpy(bo_i.map<void *>(), insts.data(), insts.size() * 4);
    bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::bo bo_b(dev, Bsh.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
    memcpy(bo_b.map<void *>(), Bsh.data(), Bsh.size());
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // XRT's own buffers
    xrt::bo a0(dev, Ash.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
    xrt::bo c0(dev, cb, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
    memcpy(a0.map<void *>(), Ash.data(), Ash.size());
    a0.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ours: page-aligned host memory, sizes rounded up to whole pages
    const size_t page = 4096;
    auto round_up = [&](size_t x) { return (x + page - 1) / page * page; };
    const size_t a_sz = round_up(Ash.size()), c_sz = round_up(cb);
    void * a_mem = _aligned_malloc(a_sz, page);
    void * c_mem = _aligned_malloc(c_sz, page);
    memcpy(a_mem, Ash.data(), Ash.size());
    xrt::bo a1, c1;  // xrt::ext::bo only adds constructors
    try {
        a1 = xrt::ext::bo(dev, a_mem, a_sz);
        c1 = xrt::ext::bo(dev, c_mem, c_sz);
    } catch (const std::exception & e) {
        printf("XRT refused a buffer over our own memory: %s\nFAIL\n", e.what());
        return 1;
    }
    printf("buffers over our memory created; map() gives our pointer back: A %s, C %s\n",
           a1.map<void *>() == a_mem ? "yes" : "no", c1.map<void *>() == c_mem ? "yes" : "no");
    a1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto make_run = [&](xrt::bo & a, xrt::bo & c) {
        xrt::run r(kern);
        r.set_arg(0, 3u);
        r.set_arg(1, bo_i);
        r.set_arg(2, (unsigned) insts.size());
        r.set_arg(3, a);
        r.set_arg(4, bo_b);
        r.set_arg(5, c);
        return r;
    };
    xrt::run r0 = make_run(a0, c0), r1 = make_run(a1, c1);
    auto dispatch = [&](xrt::run & r) {
        auto t0 = clk::now();
        r.start();
        auto st = r.wait(std::chrono::milliseconds(20000));
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        if (st != ERT_CMD_STATE_COMPLETED) { fprintf(stderr, "state %d\n", (int) st); exit(1); }
        return ms;
    };

    int bad = 0;
    for (int round = 0; round < 3; round++) {
        memset(c0.map<void *>(), 0xFF, cb);  // NaN everywhere, so a value never written can't pass
        c0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        memset(c_mem, 0xFF, cb);
        c1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        dispatch(r0);
        dispatch(r1);
        // before any sync: what the CPU sees in our memory straight away
        const bool same_unsynced = memcmp(c_mem, c0.map<void *>(), cb) == 0;
        c0.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        c1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const bool same = memcmp(c_mem, c0.map<void *>(), cb) == 0;
        // the reference, in the tiled layout's element order: compare as a set of
        // values via the same rounding-free sum check mixk uses
        std::vector<float> C0((size_t) M * N);
        memcpy(C0.data(), c0.map<void *>(), cb);
        std::vector<float> Crow((size_t) M * N);
        // the builds here are --c-tiled: undo the tiling for the reference check
        bfp16_tiling t;
        t.m = m; t.k = k; t.n = n;
        for (int rr = 0; rr < M; rr++) bfp16_c_row(C0.data(), M, N, t, rr, N, Crow.data() + (size_t) rr * N);
        const Err e = compare(Crow, ref);
        const bool ok = same && e.frob_rel < 1e-5;
        bad += !ok;
        printf("round %d: XRT buffers vs reference frob_rel %.2e | our memory: %s the XRT run's output bit for bit "
               "(%s before the output sync)\n",
               round, e.frob_rel, same ? "matches" : "DIFFERS from", same_unsynced ? "already matching" : "not yet matching");
    }

    std::vector<double> t0, t1;
    for (int i = 0; i < 5; i++) { dispatch(r0); dispatch(r1); }
    for (int i = 0; i < iters; i++) {
        t0.push_back(dispatch(r0));
        t1.push_back(dispatch(r1));
    }
    printf("%dx%dx%d, %d runs each, alternating: XRT buffers median %.4f ms, our memory median %.4f ms (%+.1f%%)\n", M, K, N,
           iters, median(t0), median(t1), 100.0 * (median(t1) / median(t0) - 1));
    printf("%s\n", bad ? "FAIL" : "PASS");
    // the memory stays allocated: the buffers over it live until exit
    return bad ? 1 : 0;
}
