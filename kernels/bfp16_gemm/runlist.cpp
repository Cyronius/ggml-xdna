// Per-run cost of xrt::runlist against individual start/wait, under ONE
// runtime-K xclbin (whole_array_bfp_rtp) and one hw context.
//
//   runlist.exe <xclbin> <iters> <dir:M:K:N:count> [<dir:M:K:N:count> ...]
//   runlist.exe <xclbin> limit <dir:M:K:N>
//
// Each group contributes <count> runs of that shape, every run with its own
// A/B/C buffers and its own random inputs (the insts BO is shared per shape).
// The runs of all groups, in the order given, form one batch. Per iteration:
//   single : start/wait each run in turn            -> total / runs
//   runlist: one xrt::runlist of all runs, execute+wait -> total / runs
// (runlist construction timed separately). Every output is checked against a
// double CPU reference after both modes.
// "limit": one shape, runlists of 8, 16, ... runs until XRT refuses one.

#define BENCH_NO_MAIN
#include "bench_bfp16.cpp"

struct Shape {
    std::string dir;
    int M, K, N;
    xrt::bo bo_i;
    unsigned ninst = 0;
};
struct Run {
    Shape * s;
    xrt::bo bo_a, bo_b, bo_c;
    xrt::run run;
    std::vector<double> ref;
};

static const int m = 128, k = 64, n = 64;

// "dir:M:K:N" or "dir:M:K:N:count"; the dir may itself contain ':' (C:\...).
static void parse(const std::string & spec, Shape & s, int * count) {
    const int nf = count ? 4 : 3;
    std::vector<int> v(nf);
    size_t end = spec.size();
    for (int i = nf - 1; i >= 0; i--) {
        size_t p = spec.rfind(':', end - 1);
        v[i] = atoi(spec.substr(p + 1, end - p - 1).c_str());
        end = p;
    }
    s.dir = spec.substr(0, end);
    s.M = v[0]; s.K = v[1]; s.N = v[2];
    if (count) *count = v[3];
}

static void load_insts(xrt::device & dev, xrt::kernel & kern, Shape & s) {
    auto insts = read_insts(s.dir + "/insts.bin");
    s.ninst = (unsigned) insts.size();
    s.bo_i = xrt::bo(dev, insts.size() * 4, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
    memcpy(s.bo_i.map<void *>(), insts.data(), insts.size() * 4);
    s.bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

static void make_run(xrt::device & dev, xrt::kernel & kern, Shape & s, Run & r, unsigned seed, bool with_ref) {
    const int M = s.M, K = s.K, N = s.N;
    std::mt19937 rng(seed);
    std::normal_distribution<float> da(0.f, 1.f), dw(0.f, 0.05f);
    std::vector<float> A((size_t) M * K), Bt((size_t) N * K);
    for (auto & x : A) x = bf2f(f2bf(da(rng)));
    for (auto & x : Bt) x = dw(rng);
    auto Abfp = floatToBfp16Rne(M * K, A.data());
    auto Bbfp = floatToBfp16(8, N * K, Bt.data(), 0);
    auto Ash = linA(M, K, m, k, Abfp), Bsh = linB(N, K, n, k, 8, Bbfp);
    if (with_ref) {
        auto Aq = bfp16ebs8ToFloat((int) Abfp.size(), Abfp.data());
        auto Bq = bfp16ebs8ToFloat((int) Bbfp.size(), Bbfp.data());
        r.ref = ref_gemm(M, K, N, Aq, Bq);
    }
    r.s = &s;
    r.bo_a = xrt::bo(dev, Ash.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
    r.bo_b = xrt::bo(dev, Bsh.size(), XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
    r.bo_c = xrt::bo(dev, (size_t) M * N * 4, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
    memcpy(r.bo_a.map<void *>(), Ash.data(), Ash.size());
    memcpy(r.bo_b.map<void *>(), Bsh.data(), Bsh.size());
    r.bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    r.bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    r.run = xrt::run(kern);
    r.run.set_arg(0, 3u);
    r.run.set_arg(1, s.bo_i);
    r.run.set_arg(2, s.ninst);
    r.run.set_arg(3, r.bo_a);
    r.run.set_arg(4, r.bo_b);
    r.run.set_arg(5, r.bo_c);
}

static void poison(std::vector<Run> & runs) {
    for (auto & r : runs) {
        memset(r.bo_c.map<void *>(), 0xFF, (size_t) r.s->M * r.s->N * 4);
        r.bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
}
static int check(std::vector<Run> & runs) {
    int bad = 0;
    for (auto & r : runs) {
        r.bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::vector<float> C((size_t) r.s->M * r.s->N);
        memcpy(C.data(), r.bo_c.map<void *>(), C.size() * 4);
        if (!(compare(C, r.ref).frob_rel < 1e-5)) bad++;
    }
    return bad;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <xclbin> <iters>|limit <dir:M:K:N[:count]>...\n", argv[0]); return 2; }
    xrt::device dev(0);
    const std::string xpath = argv[1];
    xrt::xclbin xb(xpath);
    dev.register_xclbin(xb);
    xrt::hw_context ctx(dev, xb.get_uuid());
    std::string kname;
    for (auto & kk : xb.get_kernels())
        if (kk.get_name().rfind("MLIR_AIE", 0) == 0) kname = kk.get_name();
    xrt::kernel kern(ctx, kname);

    if (std::string(argv[2]) == "limit") {
        Shape s;
        parse(argv[3], s, nullptr);
        load_insts(dev, kern, s);
        Run proto;
        make_run(dev, kern, s, proto, 7, false);
        for (int R = 8; R <= 4096; R *= 2) {
            try {
                std::vector<xrt::run> rs;
                for (int i = 0; i < R; i++) {
                    xrt::run r(kern);
                    r.set_arg(0, 3u); r.set_arg(1, s.bo_i); r.set_arg(2, s.ninst);
                    r.set_arg(3, proto.bo_a); r.set_arg(4, proto.bo_b); r.set_arg(5, proto.bo_c);
                    rs.push_back(r);
                }
                xrt::runlist rl(ctx);
                for (auto & r : rs) rl.add(r);
                auto t0 = clk::now();
                rl.execute();
                rl.wait(std::chrono::milliseconds(120000));
                double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                printf("runlist of %4d runs: ok, %.4f ms per run\n", R, ms / R);
            } catch (const std::exception & e) {
                printf("runlist of %4d runs: FAILED: %s\n", R, e.what());
                break;
            }
        }
        return 0;
    }

    const int iters = atoi(argv[2]);
    std::vector<std::unique_ptr<Shape>> shapes;
    std::vector<Run> runs;
    std::vector<int> counts;
    for (int i = 3; i < argc; i++) {
        auto s = std::make_unique<Shape>();
        int count = 0;
        parse(argv[i], *s, &count);
        load_insts(dev, kern, *s);
        counts.push_back(count);
        shapes.push_back(std::move(s));
    }
    size_t total = 0;
    for (int c : counts) total += c;
    runs.resize(total);
    for (size_t g = 0, r = 0; g < shapes.size(); g++)
        for (int i = 0; i < counts[g]; i++, r++) make_run(dev, kern, *shapes[g], runs[r], 1000 + (unsigned) r, true);

    double flop = 0;
    for (auto & r : runs) flop += 2.0 * r.s->M * r.s->K * r.s->N;
    printf("batch: %zu runs (", runs.size());
    for (size_t g = 0; g < shapes.size(); g++)
        printf("%s%d x %dx%dx%d", g ? ", " : "", counts[g], shapes[g]->M, shapes[g]->K, shapes[g]->N);
    printf("), %.2f GFLOP, own A/B/C per run\n", flop / 1e9);

    // warm up both paths once, verified
    poison(runs);
    for (auto & r : runs) { r.run.start(); r.run.wait(); }
    int bad_single = check(runs);
    poison(runs);
    { xrt::runlist rl(ctx); for (auto & r : runs) rl.add(r.run); rl.execute(); rl.wait(); }
    int bad_rl = check(runs);

    std::vector<double> t_single, t_rl, t_build;
    for (int it = 0; it < iters; it++) {
        auto t0 = clk::now();
        for (auto & r : runs) {
            r.run.start();
            if (r.run.wait(std::chrono::milliseconds(20000)) != ERT_CMD_STATE_COMPLETED) { printf("single run failed\n"); return 1; }
        }
        t_single.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count() / runs.size());

        auto tb = clk::now();
        xrt::runlist rl(ctx);
        for (auto & r : runs) rl.add(r.run);
        auto te = clk::now();
        rl.execute();
        rl.wait(std::chrono::milliseconds(120000));
        auto tw = clk::now();
        t_build.push_back(std::chrono::duration<double, std::milli>(te - tb).count());
        t_rl.push_back(std::chrono::duration<double, std::milli>(tw - te).count() / runs.size());
    }
    // verify once more after the timed iterations
    poison(runs);
    { xrt::runlist rl(ctx); for (auto & r : runs) rl.add(r.run); rl.execute(); rl.wait(); }
    bad_rl += check(runs);

    double s = median(t_single), l = median(t_rl);
    printf("single start/wait : %.4f ms per run  (batch %.3f ms, %.2f TFLOP/s)\n", s, s * runs.size(), flop / (s * runs.size()) / 1e9);
    printf("one runlist       : %.4f ms per run  (batch %.3f ms, %.2f TFLOP/s)  + %.3f ms to build the runlist\n",
           l, l * runs.size(), flop / (l * runs.size()) / 1e9, median(t_build));
    printf("saved per run     : %.4f ms   (iters %d)\n", s - l, iters);
    printf("outputs failing reference: single %d/%zu, runlist %d/%zu\n", bad_single, runs.size(), bad_rl, 2 * runs.size());
    return (bad_single || bad_rl) ? 1 : 0;
}
