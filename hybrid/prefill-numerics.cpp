// Stage 1 of .claude/plans/npu-prefill-beats-gpu.md: does rounding the prefill's
// matmul inputs the way an NPU datapath would still leave the GPU writing the
// same answer after the handoff?
//
// For each prompt, the host reference (hybrid/qwen3_ref.cpp) prefills once in
// float32 and once per rounding mode, all from the same 4-bit file, so the
// weights' own quantisation is common to every run and only the datapath
// differs. Each run's keys and values go to a GPU context, which then decodes.
// The float32 run decodes greedily and fixes the continuation; every other run
// is fed those same tokens (teacher forcing), so every step is compared
// against the float32 run's logits at the same position rather than stopping
// at the first near-tie that flips.
//
// Two yardsticks run the same way: the GPU prefilling the prompt itself, and
// the reference with llama.cpp's own CPU rounding (8-bit activations in blocks
// of 32). A datapath that disturbs the answer no more than those do is as good
// as what llama.cpp already ships.
//
// Traces: HYBRID-BFP16-NUMERICS

#include "common.h"
#include "kv_state.h"
#include "qwen3_ref.h"
#ifdef HAVE_NPU
#include "npu_prefill.h"
#include "npu_projections.h"
#endif

#include "ggml.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

namespace {

struct variant {
    const char *       name;
    qwen3_prefill_opts opts;
    bool               gpu_self = false;  // the GPU prefills the prompt itself instead of taking our state
    bool               npu = false;       // projections on the NPU (--npu)
    int                prefill = 0;       // the whole NPU prefill (--prefill): 1 with fp32 outputs, 2 with 16-bit
    int                sim = -1;          // with prefill: the variant simulating it, to compare states with
};

struct tally {
    double min_kv_cos = 1.0, max_kv_nrmse = 0.0;
    double kl_sum = 0.0, kl_max = 0.0;
    int    steps = 0, top1 = 0;
    int    first_ok = 0, prompts = 0;
};

// KL(p || q) of two logit vectors, in nats
double kl_div(const float * lp, const float * lq, int n) {
    double mp = -1e300, mq = -1e300;
    for (int i = 0; i < n; i++) { mp = std::max(mp, (double) lp[i]); mq = std::max(mq, (double) lq[i]); }
    double zp = 0, zq = 0;
    for (int i = 0; i < n; i++) { zp += std::exp(lp[i] - mp); zq += std::exp(lq[i] - mq); }
    const double lzp = std::log(zp) + mp, lzq = std::log(zq) + mq;
    double kl = 0;
    for (int i = 0; i < n; i++) {
        const double a = lp[i] - lzp;
        kl += std::exp(a) * (a - (lq[i] - lzq));
    }
    return std::max(kl, 0.0);
}

void compare_kv(const qwen3_prefill_out & a, const qwen3_prefill_out & b, double & min_cos, double & max_nrmse) {
    min_cos = 1.0; max_nrmse = 0.0;
    auto one = [&](const std::vector<uint16_t> & x, const std::vector<uint16_t> & y) {
        std::vector<float> fx(x.size()), fy(y.size());
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) x.data(), fx.data(), fx.size());
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) y.data(), fy.data(), fy.size());
        double se = 0, sy = 0, dot = 0, nx = 0;
        for (size_t i = 0; i < fx.size(); i++) {
            const double d = fx[i] - fy[i];
            se += d * d; sy += (double) fy[i] * fy[i];
            dot += (double) fx[i] * fy[i]; nx += (double) fx[i] * fx[i];
        }
        min_cos   = std::min(min_cos, dot / std::sqrt(nx * sy));
        max_nrmse = std::max(max_nrmse, std::sqrt(se / std::max(sy, 1e-300)));
    };
    for (size_t il = 0; il < a.k.size(); il++) { one(a.k[il], b.k[il]); one(a.v[il], b.v[il]); }
}

std::vector<uint8_t> state_blob(const qwen3_prefill_out & R, const qwen3_hparams & hp, int n_prompt) {
    const size_t row_el = (size_t) hp.n_head_kv * hp.head_dim;
    kv_state s;
    s.seq_id = 0;
    s.cell_count = n_prompt;
    s.pos.resize(n_prompt);
    s.seq_ids.assign(n_prompt, { 0 });
    for (int i = 0; i < n_prompt; i++) s.pos[i] = i;
    s.v_trans = false;
    s.k.resize(hp.n_layer);
    s.v.resize(hp.n_layer);
    for (int il = 0; il < hp.n_layer; il++) {
        auto fill = [&](kv_state::layer_rows & L, const std::vector<uint16_t> & src) {
            L.type = GGML_TYPE_F16;
            L.row_bytes = row_el * 2;
            L.data.resize(src.size() * 2);
            memcpy(L.data.data(), src.data(), L.data.size());
        };
        fill(s.k[il], R.k[il]);
        fill(s.v[il], R.v[il]);
    }
    return kv_state_write(s);
}

// Logits at each step: after the last prompt token, then after each of `cont`.
// With `cont` empty, decodes greedily for n steps and fills it.
std::vector<std::vector<float>> gpu_steps(llama_context * ctx, const std::vector<llama_token> & toks,
                                          const std::vector<uint8_t> * blob, std::vector<llama_token> & cont,
                                          int n_steps, int n_vocab) {
    const int n_prompt = (int) toks.size();
    llama_memory_clear(llama_get_memory(ctx), true);
    if (blob) {
        if (llama_state_seq_set_data(ctx, blob->data(), blob->size(), 0) == 0) die("state import refused");
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prompt - 1, -1)) die("seq_rm");
        decode_one(ctx, toks[n_prompt - 1], "last prompt token");
    } else {
        decode_all(ctx, toks, "gpu prefill");
    }
    const bool greedy_mode = cont.empty();
    std::vector<std::vector<float>> out;
    out.push_back(take_logits(ctx, n_vocab));
    for (int i = 0; i < n_steps; i++) {
        if (greedy_mode) cont.push_back(argmax(out.back().data(), n_vocab));
        decode_one(ctx, cont[i], "continuation");
        out.push_back(take_logits(ctx, n_vocab));
    }
    return out;
}

// The softmax_npu bounds (plan: .claude/plans/softmax-on-npu.md, step 0)
void print_bias_stats(const char * label, const softmax_bias_stats & b, float window) {
    auto pct = [](int64_t a, int64_t n) { return 100.0 * a / std::max<int64_t>(1, n); };
    // the value below which a fraction f of rows fall, in whole log2 units
    auto q = [&](const std::vector<int64_t> & h, double f) {
        int64_t n = 0, acc = 0;
        for (int64_t c : h) n += c;
        for (int i = 0; i < softmax_bias_stats::NB; i++) {
            acc += h[i];
            if (acc >= f * n) return i + 1;
        }
        return softmax_bias_stats::NB;
    };
    auto over = [&](const std::vector<int64_t> & h, int x) {
        int64_t a = 0;
        for (int i = x; i < softmax_bias_stats::NB; i++) a += h[i];
        return pct(a, b.rows);
    };
    printf("  %s: bias window %.0f; fall back: %.3f%% of rows, %.2f%% of 128-query tiles, %.2f%% of scores calls\n",
           label, window, pct(b.rows_fallback, b.rows), pct(b.tiles_fallback, b.tiles), pct(b.calls_fallback, b.calls));
    printf("    |q| max|k| the tighter upper bound on %.1f%% of rows\n", pct(b.cs_tighter, b.rows));
    const struct { const char * name; const std::vector<int64_t> & h; double worst; } rows[] = {
        { "upper bound - max", b.up_gap, b.worst_up },
        { "max - lower bound", b.lo_gap, b.worst_lo },
        { "upper - lower", b.window, b.worst_window },
    };
    printf("    %-18s  %6s %6s %6s %6s %7s   %7s %7s %7s\n", "log2 units", "p50", "p90", "p99", "p99.9", "worst", ">60", ">120",
           ">180");
    for (const auto & r : rows)
        printf("    %-18s  %6d %6d %6d %6d %7.1f   %6.3f%% %6.3f%% %6.3f%%\n", r.name, q(r.h, 0.5), q(r.h, 0.9), q(r.h, 0.99),
               q(r.h, 0.999), r.worst, over(r.h, 60), over(r.h, 120), over(r.h, 180));
}

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::vector<std::string> prompt_files;
    int n_steps = 32, n_ctx = 4096;
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    bool quick = false, ablate = false, accum = false, attn = false, outputs = false, softmax = false;
    std::string npu_root;
    std::string prefill_root;  // --prefill: kernel builds for npu_prefill (hybrid/npu_prefill.h)

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--prompt-file" && i + 1 < argc) prompt_files.push_back(argv[++i]);
        else if (a == "--steps"       && i + 1 < argc) n_steps   = atoi(argv[++i]);
        else if (a == "--threads"     && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (a == "--quick")                       quick     = true;
        else if (a == "--ablate")                      ablate    = true;
        else if (a == "--accum")                       accum     = true;
        else if (a == "--attn")                        attn      = true;
        else if (a == "--outputs")                     outputs   = true;
        else if (a == "--softmax")                     softmax   = true;
        else if (a == "--npu"         && i + 1 < argc) npu_root  = argv[++i];
        else if (a == "--prefill"     && i + 1 < argc) prefill_root = argv[++i];
        else if (model_path.empty())                   model_path = a;
        else die("unexpected argument " + a);
    }
    if (model_path.empty() || prompt_files.empty()) {
        die("usage: prefill-numerics <qwen3-q4.gguf> --prompt-file f [--prompt-file f ...] [--steps n] [--quick]");
    }

    std::vector<variant> variants = {
        { "float32 (oracle)",      { qwen3_rounding::none,        false } },
        { "GPU prefills itself",   { qwen3_rounding::none,        false }, true },
        { "q8_0 acts (llama CPU)", { qwen3_rounding::q8_0,        false } },
        { "bf16 (today's NPU)",    { qwen3_rounding::bf16,        false } },
        { "bfp16 nearest",         { qwen3_rounding::bfp16,       false } },
        { "bfp16 floor (in-core)", { qwen3_rounding::bfp16_floor, false } },
        { "bfp16 floor + attn",    { qwen3_rounding::bfp16_floor, true  } },
        { "bf16 + attn",           { qwen3_rounding::bf16,        true  } },
    };
    if (quick) variants.resize(6);
    if (ablate) {
        // where bfp16-nearest's error comes from
        using O = qwen3_prefill_opts;
        auto v = [](qwen3_rounding r, bool kw, bool ka, unsigned bf) {
            qwen3_prefill_opts o; o.rounding = r; o.keep_weights = kw; o.keep_acts = ka; o.bf16_projections = bf; return o;
        };
        variants = {
            { "float32 (oracle)",      v(qwen3_rounding::none,  false, false, 0) },
            { "GPU prefills itself",   v(qwen3_rounding::none,  false, false, 0), true },
            { "bfp16 nearest",         v(qwen3_rounding::bfp16, false, false, 0) },
            { "  weights only",        v(qwen3_rounding::bfp16, false, true,  0) },
            { "  activations only",    v(qwen3_rounding::bfp16, true,  false, 0) },
            { "  qkv kept bf16",       v(qwen3_rounding::bfp16, false, false, O::QKV) },
            { "  o kept bf16",         v(qwen3_rounding::bfp16, false, false, O::O) },
            { "  gate/up kept bf16",   v(qwen3_rounding::bfp16, false, false, O::GATE_UP) },
            { "  down kept bf16",      v(qwen3_rounding::bfp16, false, false, O::DOWN) },
        };
    }

    if (accum) {
        // what the kernel's running-sum precision does on top of bfp16-nearest inputs
        using A = qwen3_prefill_opts::acc_type;
        auto v = [](A acc, int kc) {
            qwen3_prefill_opts o; o.rounding = qwen3_rounding::bfp16; o.acc = acc; o.acc_chunk = kc; return o;
        };
        variants = {
            { "float32 (oracle)",      {} },
            { "GPU prefills itself",   {}, true },
            { "bfp16, f32 sums",       v(A::f32, 64) },
            { "  bf16 sums every 256", v(A::bf16, 256) },
            { "  bf16 sums every 64",  v(A::bf16, 64) },
            { "  bfp16 sums every 128",v(A::bfp16, 128) },
            { "  bfp16 sums every 64", v(A::bfp16, 64) },
        };
    }

    if (attn) {
        // can attention's two products also run in bfp16 (rounded to nearest, float32 sums)?
        auto v = [](qwen3_rounding r, bool ra, bool sc = true, bool va = true, bool split = false) {
            qwen3_prefill_opts o; o.rounding = r; o.round_attention = ra; o.attn_scores = sc; o.attn_values = va;
            o.attn_scores_split = split; return o;
        };
        variants = {
            { "float32 (oracle)",      {} },
            { "GPU prefills itself",   {}, true },
            { "bfp16 nearest",         v(qwen3_rounding::bfp16, false) },
            { "  + values",            v(qwen3_rounding::bfp16, true, false, true) },
            { "  + values, split scores", v(qwen3_rounding::bfp16, true, true, true, true) },
            { "  + split scores only", v(qwen3_rounding::bfp16, true, true, false, true) },
        };
    }

    if (outputs) {
        // what 16-bit NPU outputs cost on top of today's NPU path (plan:
        // .claude/plans/smaller-npu-outputs.md, step 0). Today's path:
        // projections in bfp16, scores as the hi/lo split, values in bfp16.
        using O = qwen3_prefill_opts;
        auto today = [] {
            qwen3_prefill_opts o; o.rounding = qwen3_rounding::bfp16; o.round_attention = true;
            o.attn_scores_split = true; return o;
        };
        qwen3_prefill_opts planned = today(), scores = today(), attn_out = today();
        planned.out_bf16 = O::QKV | O::O | O::DOWN;
        planned.swiglu_bf16 = true;
        scores = planned;   scores.scores_bf16 = true;
        attn_out = planned; attn_out.attn_out_bf16 = true;
        variants = {
            { "float32 (oracle)",       {} },
            { "GPU prefills itself",    {}, true },
            { "NPU today (simulated)",  today() },
            { "+ 16-bit outputs, SiLU", planned },
            { "  + scores 16-bit",      scores },
            { "  + attn output 16-bit", attn_out },
        };
    }

    if (softmax) {
        // attention's softmax the way the NPU would do it, on top of today's
        // NPU path with 16-bit outputs (plan: .claude/plans/softmax-on-npu.md,
        // step 0)
        using O = qwen3_prefill_opts;
        qwen3_prefill_opts today;
        today.rounding = qwen3_rounding::bfp16;
        today.round_attention = true;
        today.attn_scores_split = true;
        today.out_bf16 = O::QKV | O::O | O::DOWN;
        today.swiglu_bf16 = true;
        qwen3_prefill_opts npu_sm = today, npu_sm_bf16 = today;
        npu_sm.softmax_npu = npu_sm_bf16.softmax_npu = true;
        npu_sm_bf16.exp_bf16 = true;
        variants = {
            { "float32 (oracle)",       {} },
            { "GPU prefills itself",    {}, true },
            { "NPU today (simulated)",  today },
            { "+ softmax on the NPU",   npu_sm },
            { "  exp rounded to bf16",  npu_sm_bf16 },
        };
    }

    if (!prefill_root.empty()) {
        // the whole NPU prefill (hybrid/npu_prefill.h), with 32-bit and with
        // 16-bit projection outputs, each against its simulation (plan:
        // .claude/plans/smaller-npu-outputs.md, step 4)
        using O = qwen3_prefill_opts;
        qwen3_prefill_opts today;
        today.rounding = qwen3_rounding::bfp16;
        today.round_attention = true;
        today.attn_scores_split = true;
        qwen3_prefill_opts planned = today;
        planned.out_bf16 = O::QKV | O::O | O::DOWN;
        planned.swiglu_bf16 = true;
        variants = {
            { "float32 (oracle)",       {} },
            { "GPU prefills itself",    {}, true },
            { "NPU 32-bit, simulated",  today },
            { "NPU 16-bit, simulated",  planned },
            { "NPU 32-bit outputs",     {}, false, false, 1, 2 },
            { "NPU 16-bit outputs",     {}, false, false, 2, 3 },
        };
    }

    if (!npu_root.empty()) {
        // the real kernel against the simulation of it: projections in bfp16,
        // attention in float32 on the host in both
        qwen3_prefill_opts sim;
        sim.rounding = qwen3_rounding::bfp16;
        variants = {
            { "float32 (oracle)",      {} },
            { "GPU prefills itself",   {}, true },
            { "bfp16 simulated",       sim },
            { "bfp16 on the NPU",      {}, false, true },
        };
    }

    ggml_backend_load_all();
    llama_backend_init();
    quiet_llama_logs();

    ggml_backend_dev_t gpu = find_gpu();
    if (!gpu) die("no GPU device registered");
    ggml_backend_dev_t gpus[] = { gpu, nullptr };
    llama_model_params mp = llama_model_default_params();
    mp.devices = gpus;
    mp.n_gpu_layers = 999;
    llama_model * m_gpu = llama_model_load_from_file(model_path.c_str(), mp);
    if (!m_gpu) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(m_gpu);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = cp.n_ubatch = 2048;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(m_gpu, cp);
    if (!ctx) die("context");

    qwen3_ref ref;
    std::string err;
    if (!ref.load(model_path, err)) die("reference load: " + err);
    const qwen3_hparams & hp = ref.hp();

    printf("model: %s\nsteps per prompt: %d, threads: %d\n\n", model_path.c_str(), n_steps, n_threads);

#ifdef HAVE_NPU
    npu_projections npu;
    if (!npu_root.empty()) {
        clk::time_point t0 = clk::now();
        if (!npu.init(ref, npu_root, n_threads, err)) die("NPU: " + err);
        printf("NPU: every projection packed and resident in %.1f s\n\n", ms_since(t0) / 1000.0);
    }
    npu_prefill full32, full16;
    if (!prefill_root.empty()) {
        clk::time_point t0 = clk::now();
        full32.out16 = false;
        if (!full32.load(ref, prefill_root, n_threads, err) || !full16.load(ref, prefill_root, n_threads, err))
            die("NPU prefill: " + err);
        if (!full16.out16_used()) die("no 16-bit output builds in " + prefill_root);
        printf("NPU prefill: loaded twice (32-bit and 16-bit outputs) in %.1f s\n\n", ms_since(t0) / 1000.0);
    }
#else
    if (!npu_root.empty() || !prefill_root.empty()) die("built without the NPU driver");
#endif

    std::vector<tally> T(variants.size());
    softmax_bias_stats bias_all;
    for (const std::string & pf : prompt_files) {
        const std::vector<llama_token> toks = tokenize(vocab, read_file(pf));
        const int n_prompt = (int) toks.size();
        if (n_prompt < 2 || n_prompt + n_steps + 1 > n_ctx) die("prompt length out of range: " + pf);
        printf("prompt %s: %d tokens\n", pf.c_str(), n_prompt);
        const std::vector<int32_t> ids(toks.begin(), toks.end());

        qwen3_prefill_out R0, R_sim;
        std::vector<qwen3_prefill_out> R_all(variants.size());
        std::vector<llama_token> cont;
        std::vector<std::vector<float>> L0;
        for (size_t vi = 0; vi < variants.size(); vi++) {
            const variant & V = variants[vi];
            tally & t = T[vi];
            clk::time_point t0 = clk::now();
            std::vector<std::vector<float>> L;
            qwen3_prefill_out R;
            if (V.gpu_self) {
                L = gpu_steps(ctx, toks, nullptr, cont, n_steps, n_vocab);
#ifdef HAVE_NPU
            } else if (V.prefill) {
                npu_prefill_stats st;
                if (!(V.prefill == 2 ? full16 : full32).prefill(ids, R, st, err)) die("NPU prefill: " + err);
                double c, e;
                compare_kv(R, R_all.at(V.sim), c, e);
                printf("  %s against its simulation: worst KV cosine %.6f, worst KV error %.2e\n", V.name, c, e);
                const std::vector<uint8_t> blob = state_blob(R, hp, n_prompt);
                L = gpu_steps(ctx, toks, &blob, cont, n_steps, n_vocab);
#endif
            } else {
                qwen3_prefill_opts opts = V.opts;
                softmax_bias_stats bias;  // counted once, on the variant with the exact exp
                if (opts.softmax_npu && !opts.exp_bf16) opts.bias_stats = &bias;
#ifdef HAVE_NPU
                if (V.npu) {
                    npu.npu_ms = npu.host_ms = 0;
                    npu.dispatches = 0;
                    opts.projection = [&](int l, unsigned w, const float * x, int64_t Tn, int64_t K, float * y, int64_t N) {
                        return npu(l, w, x, Tn, K, y, N);
                    };
                }
#endif
                if (!ref.prefill(ids, R, n_threads, opts)) {
#ifdef HAVE_NPU
                    die("NPU projection failed: " + npu.error());
#else
                    die("prefill failed");
#endif
                }
#ifdef HAVE_NPU
                if (V.npu) {
                    printf("  NPU: %d dispatches, %.0f ms in the kernel, %.0f ms packing and copying\n", npu.dispatches,
                           npu.npu_ms, npu.host_ms);
                    if (!R_sim.k.empty()) {
                        double c, e;
                        compare_kv(R, R_sim, c, e);
                        printf("  NPU against the simulation: worst KV cosine %.6f, worst KV error %.2e\n", c, e);
                    }
                }
#endif
                if (opts.bias_stats) {
                    print_bias_stats("this prompt", bias, opts.bias_window);
                    bias_all.merge(bias);
                }
                if (!V.npu && V.opts.rounding == qwen3_rounding::bfp16 && !V.opts.round_attention) R_sim = R;
                const std::vector<uint8_t> blob = state_blob(R, hp, n_prompt);
                L = gpu_steps(ctx, toks, &blob, cont, n_steps, n_vocab);
            }
            R_all[vi] = R;
            if (vi == 0) { R0 = R; L0 = L; }
            else if (!V.gpu_self) {
                double c, e;
                compare_kv(R, R0, c, e);
                t.min_kv_cos = std::min(t.min_kv_cos, c);
                t.max_kv_nrmse = std::max(t.max_kv_nrmse, e);
            }
            int top1 = 0;
            double kl_sum = 0, kl_max = 0;
            for (size_t s = 0; s < L.size(); s++) {
                top1 += argmax(L[s].data(), n_vocab) == argmax(L0[s].data(), n_vocab);
                const double kl = kl_div(L0[s].data(), L[s].data(), n_vocab);
                kl_sum += kl;
                kl_max = std::max(kl_max, kl);
            }
            t.first_ok += argmax(L[0].data(), n_vocab) == argmax(L0[0].data(), n_vocab);
            t.top1 += top1;
            t.steps += (int) L.size();
            t.kl_sum += kl_sum;
            t.kl_max = std::max(t.kl_max, kl_max);
            t.prompts++;
            printf("  %-22s top-1 %2d/%2zu  mean KL %.2e  max KL %.2e  (%.0f s)\n", V.name, top1, L.size(),
                   kl_sum / L.size(), kl_max, ms_since(t0) / 1000.0);
            fflush(stdout);
        }
        printf("  continuation: ");
        print_pieces(vocab, cont);
        printf("\n\n");
    }

    printf("summary over %zu prompts, %d positions each (the prompt's next token plus %d continuation steps)\n",
           prompt_files.size(), n_steps + 1, n_steps);
    printf("%-22s  %-9s  %-10s  %-10s  %-10s  %-12s  %-11s\n", "datapath", "first ok", "top-1 %", "mean KL", "max KL",
           "worst KV cos", "worst KV err");
    for (size_t vi = 0; vi < variants.size(); vi++) {
        const tally & t = T[vi];
        char cos_s[32] = "-", err_s[32] = "-";
        if (vi > 0 && !variants[vi].gpu_self) {
            snprintf(cos_s, sizeof(cos_s), "%.6f", t.min_kv_cos);
            snprintf(err_s, sizeof(err_s), "%.2e", t.max_kv_nrmse);
        }
        printf("%-22s  %4d/%-4d  %8.2f    %.2e    %.2e    %-12s  %-11s\n", variants[vi].name, t.first_ok, t.prompts,
               100.0 * t.top1 / std::max(1, t.steps), t.kl_sum / std::max(1, t.steps), t.kl_max, cos_s, err_s);
    }

    if (bias_all.rows) {
        printf("\n");
        print_bias_stats("all prompts", bias_all, qwen3_prefill_opts{}.bias_window);
    }

    llama_free(ctx);
    llama_model_free(m_gpu);
    llama_backend_free();
    return 0;
}
