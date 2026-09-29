// Stage 3e of .claude/plans/npu-prefill-beats-gpu.md: NPU prefill against the
// GPU's, same model file, same prompt lengths, one process, runs alternating.
//
// For each length: warm-up runs of both are discarded (the GPU's first
// prefill in a process compiles shaders; the NPU's first dispatches are slow),
// then `iters` rounds of GPU prefill and NPU prefill in turn. Medians are
// reported, as HYBRID-PREFILL-BASELINE requires.
//
// Correctness at each length: the GPU prefills the prompt itself and decodes
// greedily. Then the NPU's keys and values go into the GPU context, which runs
// the last prompt token and is fed that same continuation. At every step the
// two next-token distributions are compared: whether the top choice agrees,
// and the KL divergence. (Comparing greedy continuations instead breaks at
// the first near-tie that rounding tips the other way, and says nothing
// after it.)
//
// Traces: HYBRID-PREFILL-BEATS-GPU (proposed)

#include "common.h"
#include "gpu_attention.h"
#include "kv_state.h"
#include "npu_prefill.h"
#include "qwen3_ref.h"

#include "ggml.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

namespace {

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
            L.data.assign((const uint8_t *) src.data(), (const uint8_t *) src.data() + (size_t) n_prompt * row_el * 2);
        };
        fill(s.k[il], R.k[il]);
        fill(s.v[il], R.v[il]);
    }
    return kv_state_write(s);
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// log-softmax of the context's current logits
std::vector<double> log_probs(llama_context * ctx, int n_vocab) {
    const float * z = llama_get_logits_ith(ctx, -1);
    const double mx = *std::max_element(z, z + n_vocab);
    double sum = 0;
    for (int i = 0; i < n_vocab; i++) sum += std::exp(z[i] - mx);
    const double lse = mx + std::log(sum);
    std::vector<double> lp(n_vocab);
    for (int i = 0; i < n_vocab; i++) lp[i] = z[i] - lse;
    return lp;
}

double kl(const std::vector<double> & p, const std::vector<double> & q) {
    double s = 0;
    for (size_t i = 0; i < p.size(); i++) s += std::exp(p[i]) * (p[i] - q[i]);
    return s;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, npu_root, text_path = "../../hybrid/prompts/long.txt";
    std::vector<int> lens = { 264, 515, 1012, 1999 };
    int iters = 7, n_cont = 16;
    int n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    std::string attn = "npu";  // --attn npu|gpu|cpu: where attention runs
    bool out32 = false;  // every projection's output in fp32, even where 16-bit builds exist
    int streams = 2;
    bool gpu_attn_only = false;  // --gpu-attn-only: time attention on the GPU alone, nothing else running
    int gpu_gap_ms = 0;          // with it, --gpu-gap-ms: the GPU left idle this long after each layer
    std::string dump_dir;  // --dump-prompts: write each length's prompt as text there, for prefill-numerics
    // --vs-npu / --vs-out32 / --vs-streams: a second NPU configuration, run in
    // the same rounds as the first (see "Rounds" below)
    std::string vs_root, vs_attn;
    bool vs = false, vs_out32 = false;
    int vs_streams = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--npu"     && i + 1 < argc) npu_root  = argv[++i];
        else if (a == "--text"    && i + 1 < argc) text_path = argv[++i];
        else if (a == "--iters"   && i + 1 < argc) iters     = atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (a == "--one-context")             {}  // always, now; accepted for old command lines
        else if (a == "--streams" && i + 1 < argc) streams   = atoi(argv[++i]);
        else if (a == "--attn"    && i + 1 < argc) attn = argv[++i];
        else if (a == "--host-attention")          attn = "cpu";  // old command lines
        else if (a == "--out32")                   out32 = true;
        else if (a == "--vs-npu"  && i + 1 < argc) { vs = true; vs_root = argv[++i]; }
        else if (a == "--vs-out32")                { vs = true; vs_out32 = true; }
        else if (a == "--vs-streams" && i + 1 < argc) { vs = true; vs_streams = atoi(argv[++i]); }
        else if (a == "--vs-attn" && i + 1 < argc) { vs = true; vs_attn = argv[++i]; }
        else if (a == "--dump-prompts" && i + 1 < argc) dump_dir = argv[++i];
        else if (a == "--gpu-attn-only")           gpu_attn_only = true;
        else if (a == "--gpu-gap-ms" && i + 1 < argc) gpu_gap_ms = atoi(argv[++i]);
        else if (a == "--lens"    && i + 1 < argc) {
            lens.clear();
            std::stringstream ss(argv[++i]);
            std::string item;
            while (std::getline(ss, item, ',')) lens.push_back(atoi(item.c_str()));
        }
        else if (model_path.empty()) model_path = a;
        else die("unexpected argument " + a);
    }
    if (model_path.empty() || npu_root.empty()) {
        die("usage: npu-prefill-bench <qwen3-q4.gguf> --npu <kernel builds> [--lens 264,515,...] [--iters n] [--text f]\n"
            "       [--streams n] [--out32] [--attn npu|gpu|cpu]\n"
            "       [--vs-npu <kernel builds>] [--vs-out32] [--vs-streams n] [--vs-attn npu|gpu|cpu]");
    }

    setvbuf(stdout, nullptr, _IONBF, 0);  // progress shows even when piped
    ggml_backend_load_all();
    llama_backend_init();
    quiet_llama_logs();
    ggml_backend_dev_t gpu = find_gpu();
    if (!gpu) die("no GPU device registered");
    ggml_backend_dev_t gpus[] = { gpu, nullptr };
    llama_model_params mp = llama_model_default_params();
    mp.devices = gpus;
    mp.n_gpu_layers = 999;
    llama_model * m = llama_model_load_from_file(model_path.c_str(), mp);
    if (!m) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int max_len = *std::max_element(lens.begin(), lens.end());

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = max_len + n_cont + 64;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;  // llama-bench's default
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) die("context");

    // prompt tokens: the text, repeated as often as the longest length needs
    std::vector<llama_token> all;
    {
        std::ifstream f(text_path, std::ios::binary);
        if (!f) die("cannot read " + text_path);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string text = ss.str(), acc;
        while ((int) all.size() < max_len) {
            acc += text + "\n";
            all = tokenize(vocab, acc);
        }
    }

    qwen3_ref ref;
    std::string err;
    if (!ref.load(model_path, err)) die("reference load: " + err);
    if (gpu_attn_only) {
        // Attention alone on the GPU, as a prefill would run it: every layer,
        // each stream's rows against the keys before them, the streams split
        // as npu_prefill splits them. The values don't matter to the timing;
        // they're arbitrary but fixed.
        const qwen3_hparams & hp = ref.hp();
        const int64_t Eq = (int64_t) hp.n_head * hp.head_dim, Ek = (int64_t) hp.n_head_kv * hp.head_dim;
        gpu_attention ga;
        if (!ga.init(gpu, hp.n_head, hp.n_head_kv, hp.head_dim, err)) die("GPU attention: " + err);
        printf("attention alone on the GPU, %d layers, %d streams at most, %d timed runs after 2 warm-up runs\n",
               hp.n_layer, streams, iters);
        printf("%6s  %9s  |  %9s %9s %9s  (medians, ms)\n", "tokens", "total", "copy in", "run", "copy out");
        for (int len : lens) {
            std::vector<float> q((size_t) (len * Eq)), out((size_t) (len * Eq));
            std::vector<uint16_t> k((size_t) (len * Ek)), v((size_t) (len * Ek));
            uint32_t r = 12345;
            auto rnd = [&] { r = r * 1664525u + 1013904223u; return ((r >> 8) / 16777216.0f - 0.5f); };
            for (float & x : q) x = rnd();
            for (uint16_t & x : k) x = ggml_fp32_to_fp16(rnd());
            for (uint16_t & x : v) x = ggml_fp32_to_fp16(rnd());
            const int64_t B = npu_prefill::BLOCK, nb = (len + B - 1) / B;
            const int S = (int) std::max<int64_t>(1, std::min<int64_t>(streams, nb));
            std::vector<double> tot, tin, trun, tout;
            for (int it = 0; it < iters + 2; it++) {
                ga.take_times();
                const clk::time_point t0 = clk::now();
                for (int il = 0; il < hp.n_layer; il++) {
                    std::vector<int> tickets;
                    for (int i = 0; i < S; i++) {
                        const int64_t a = std::min<int64_t>(len, nb * i / S * B), b = std::min<int64_t>(len, nb * (i + 1) / S * B);
                        tickets.push_back(ga.start(q.data(), k.data(), v.data(), a, b, out.data()));
                    }
                    for (int t : tickets)
                        if (!ga.wait(t, err)) die("GPU attention: " + err);
                    if (gpu_gap_ms) std::this_thread::sleep_for(std::chrono::milliseconds(gpu_gap_ms));
                }
                const double ms = ms_since(t0) - (double) gpu_gap_ms * hp.n_layer;
                const gpu_attention::times g = ga.take_times();
                if (it < 2) continue;
                tot.push_back(ms);
                tin.push_back(g.in_ms);
                trun.push_back(g.run_ms);
                tout.push_back(g.out_ms);
            }
            printf("%6d  %9.1f  |  %9.1f %9.1f %9.1f%s\n", len, median(tot), median(tin), median(trun), median(tout),
                   ga.out_imported() ? "" : "  (output not imported: read uncached)");
        }
        return 0;
    }
    if (!dump_dir.empty()) {
        for (int len : lens) {
            const std::string path = dump_dir + "/prompt-" + std::to_string(len) + ".txt";
            FILE * f = fopen(path.c_str(), "wb");
            if (!f) die("cannot write " + path);
            for (int i = 0; i < len; i++) {
                char buf[256];
                const int n = llama_token_to_piece(vocab, all[i], buf, sizeof(buf), 0, true);
                if (n > 0) fwrite(buf, 1, n, f);
            }
            fclose(f);
            printf("wrote %s\n", path.c_str());
        }
        return 0;
    }

    // The NPU configurations: A from --npu / --out32 / --streams / --attn,
    // and with any --vs-* option a B that differs only in what those change.
    struct config {
        char name;
        std::string root;
        bool out32;
        int streams;
        std::string attn;
        std::unique_ptr<npu_prefill> npu;
    };
    auto attn_of = [](const std::string & a) {
        if (a == "npu") return npu_prefill::attn_where::npu;
        if (a == "gpu") return npu_prefill::attn_where::gpu;
        if (a == "cpu") return npu_prefill::attn_where::cpu;
        die("--attn takes npu, gpu or cpu, not " + a);
    };
    std::vector<config> cfg;
    cfg.push_back({ 'A', npu_root, out32, streams, attn, nullptr });
    if (vs)
        cfg.push_back({ 'B', vs_root.empty() ? npu_root : vs_root, vs_out32, vs_streams ? vs_streams : streams,
                        vs_attn.empty() ? attn : vs_attn, nullptr });
    for (config & c : cfg) {
        clk::time_point t0 = clk::now();
        c.npu = std::make_unique<npu_prefill>();
        c.npu->out16 = !c.out32;
        if (!c.npu->load(ref, c.root, n_threads, err)) die("NPU: " + err);
        c.npu->attn = attn_of(c.attn);
        c.npu->gpu_device = gpu;
        c.npu->max_streams = c.streams;
        printf("NPU %c: %s, %d streams, projection outputs %s, attention on the %s; weights packed and resident in %.1f s, "
               "%d host threads\n",
               c.name, c.root.c_str(), c.streams, c.npu->out16_used() ? "16-bit, SiLU(gate) * up on the NPU" : "fp32",
               c.attn.c_str(), ms_since(t0) / 1000.0, n_threads);
    }
    printf("\n");

    auto gpu_prefill = [&](const std::vector<llama_token> & toks) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const clk::time_point t0 = clk::now();
        decode_all(ctx, toks, "gpu prefill");
        llama_synchronize(ctx);  // decode returns before the GPU has finished
        return ms_since(t0);
    };
    auto npu_prefill_once = [&](config & c, const std::vector<llama_token> & toks, qwen3_prefill_out & out,
                                npu_prefill_stats & st) {
        if (!c.npu->prefill(std::vector<int32_t>(toks.begin(), toks.end()), out, st, err)) die("NPU prefill: " + err);
        return st.total_ms;
    };

    // Of each NPU configuration: time the host spent waiting for the NPU,
    // then its own work (norms, reading outputs, attention), syncs (inside
    // those), NPU calls and streams.
    printf("%6s  %9s %8s  %9s %8s  %6s  |  %6s %6s %6s %6s %6s %5s %3s  | first  top-1  KL mean / max  | cfg\n", "tokens",
           "GPU ms", "tok/s", "NPU ms", "tok/s", "GPU/NPU", "wait", "norm", "read", "attn", "sync", "calls", "str");
    for (int len : lens) {
        const std::vector<llama_token> toks(all.begin(), all.begin() + len);
        const size_t nc = cfg.size();
        std::vector<qwen3_prefill_out> out(nc);
        npu_prefill_stats st;

        for (int i = 0; i < 3; i++) gpu_prefill(toks);
        for (size_t c = 0; c < nc; c++)
            for (int i = 0; i < 2; i++) npu_prefill_once(cfg[c], toks, out[c], st);

        // Rounds: the GPU, then every configuration, the order of the
        // configurations reversed every other round. Load that comes and goes
        // then lands on all of them alike, and each round's B / A ratio is a
        // paired comparison that slow drift cancels out of.
        std::vector<double> tg;
        std::vector<std::vector<double>> tn(nc);
        std::vector<std::vector<npu_prefill_stats>> sts(nc);
        for (int i = 0; i < iters; i++) {
            tg.push_back(gpu_prefill(toks));
            for (size_t k = 0; k < nc; k++) {
                const size_t c = i % 2 ? nc - 1 - k : k;
                tn[c].push_back(npu_prefill_once(cfg[c], toks, out[c], st));
                sts[c].push_back(st);
            }
        }
        const double mg = median(tg);

        // correctness: the GPU on its own, then each configuration's state
        // fed the same tokens
        gpu_prefill(toks);
        std::vector<llama_token> self;
        std::vector<std::vector<double>> self_lp;
        for (int i = 0; i < n_cont; i++) {
            self_lp.push_back(log_probs(ctx, n_vocab));
            self.push_back(argmax(llama_get_logits_ith(ctx, -1), n_vocab));
            if (i + 1 < n_cont) decode_one(ctx, self.back(), "continuation");
        }
        for (size_t c = 0; c < nc; c++) {
            llama_memory_clear(llama_get_memory(ctx), true);
            const std::vector<uint8_t> blob = state_blob(out[c], ref.hp(), len);
            if (llama_state_seq_set_data(ctx, blob.data(), blob.size(), 0) == 0) die("state import refused");
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, len - 1, -1)) die("seq_rm");
            decode_one(ctx, toks[len - 1], "last prompt token");
            int agree = 0;
            double kl_sum = 0, kl_max = 0;
            bool first_same = false;
            for (int i = 0; i < n_cont; i++) {
                const bool same = argmax(llama_get_logits_ith(ctx, -1), n_vocab) == self[i];
                if (i == 0) first_same = same;
                agree += same;
                const double d = kl(self_lp[i], log_probs(ctx, n_vocab));
                kl_sum += d;
                kl_max = std::max(kl_max, d);
                if (i + 1 < n_cont) decode_one(ctx, self[i], "continuation, fed");
            }

            // the breakdown of the median run
            const double mn = median(tn[c]);
            const npu_prefill_stats & s = *std::min_element(sts[c].begin(), sts[c].end(), [&](const auto & a, const auto & b) {
                return std::fabs(a.total_ms - mn) < std::fabs(b.total_ms - mn);
            });
            printf("%6d  %9.1f %8.0f  %9.1f %8.0f  %6.2fx  |  %6.1f %6.1f %6.1f %6.1f %6.1f %5d %3d  | %s  %2d/%d  %.1e / %.1e"
                   "  | %c\n",
                   len, mg, len * 1000.0 / mg, mn, len * 1000.0 / mn, mg / mn, s.wait_ms, s.norm_ms, s.read_ms,
                   s.attn_in_ms + s.attn_soft_ms + s.attn_out_ms, s.sync_ms, s.calls, s.streams,
                   first_same ? "same" : "DIFF", agree, n_cont, kl_sum / n_cont, kl_max, cfg[c].name);
            printf("%6s  attention host: operands %.1f, softmax and values %.1f, output %.1f ms\n", "", s.attn_in_ms,
                   s.attn_soft_ms, s.attn_out_ms);
            if (cfg[c].attn == "gpu")
                printf("%6s  attention GPU: host waiting %.1f ms | GPU worker: copy in %.1f, run %.1f, copy out %.1f, "
                       "graph builds %.1f ms%s\n",
                       "", s.gpu_wait_ms, s.gpu_in_ms, s.gpu_run_ms, s.gpu_out_ms, s.gpu_build_ms,
                       s.gpu_out_imported ? "" : " (output not imported: read uncached)");
            // the host's time, all of it: whatever no column covers is "rest"
            const double covered = s.wait_ms + s.norm_ms + s.read_ms + s.attn_in_ms + s.attn_soft_ms + s.attn_out_ms +
                                   s.prep_ms + s.start_ms + s.other_ms + s.gpu_wait_ms;
            printf("%6s  read: q/k/v %.1f, o %.1f, gate/up %.1f, down %.1f | sync: in %.1f, out %.1f | waiting less its "
                   "syncs %.1f | prepare %.1f, start %.1f, embedding %.1f, rest %.1f ms\n",
                   "", s.read_qkv_ms, s.read_o_ms, s.read_gu_ms, s.read_down_ms, s.sync_in_ms, s.sync_out_ms,
                   s.wait_ms - s.sync_out_ms, s.prep_ms, s.start_ms, s.other_ms, s.total_ms - covered);
        }
        if (nc == 2) {
            // B's time over A's, round by round
            std::vector<double> r;
            for (int i = 0; i < iters; i++) r.push_back(tn[1][i] / tn[0][i]);
            std::sort(r.begin(), r.end());
            printf("%6s  B / A, paired by round: median %.3f, range %.3f to %.3f over %d rounds\n", "", median(r), r.front(),
                   r.back(), iters);
        }
        fflush(stdout);
    }

    llama_free(ctx);
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
