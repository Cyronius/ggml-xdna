// A trustworthy GPU prefill number for this box.
//
// Why this exists: an earlier reading of "about 1850 tokens per second" for
// GPU prefill was withdrawn because it came from one prefill in one process
// launch, and repeating that measurement in fresh launches gave readings that
// disagreed by more than ten times. A single sample cannot be quoted.
//
// So: one process, one context, prefill the same prompt many times over,
// throw the first several away, and report the spread of what is left. The
// discarded runs, and with --samples every kept run, are printed, so the shape
// of the warm-up is visible rather than assumed.
//
// Two separate things make a single reading untrustworthy here. Some launches
// compile the Vulkan shader pipelines before the first prefill can run,
// costing a second or more; the driver caches them, and which launch pays is
// not something we can predict from outside. Separately, the first few
// prefills after that still run slow while clocks ramp, and anything else
// using the GPU adds spikes of a few hundred milliseconds. The median over
// enough runs survives both. The mean does not, and one sample survives
// nothing.
//
// Decode is measured after each prefill as well, because decode was already
// steady. It doubles as a check that the box is in the same state it was in
// when the earlier figures were taken.
//
// Traces: HYBRID-PREFILL-BASELINE

#include "common.h"

#include <numeric>
#include <thread>

namespace {

struct stats {
    double mean = 0, median = 0, lo = 0, hi = 0, sd = 0;
};

stats summarise(std::vector<double> v) {
    stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.lo     = v.front();
    s.hi     = v.back();
    s.median = v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    s.mean   = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    if (v.size() > 1) {
        double acc = 0;
        for (double x : v) acc += (x - s.mean) * (x - s.mean);
        s.sd = std::sqrt(acc / (v.size() - 1));
    }
    return s;
}

// A token sequence of exactly n tokens: tokenize the text, tile it until it
// is long enough, then cut. What the tokens say does not affect timing.
std::vector<llama_token> prompt_of_length(const llama_vocab * vocab, const std::string & text, int n) {
    const std::vector<llama_token> base = tokenize(vocab, text);
    if (base.empty()) die("empty prompt");
    std::vector<llama_token> out;
    out.reserve(n);
    while ((int) out.size() < n) {
        for (llama_token t : base) {
            if ((int) out.size() == n) break;
            out.push_back(t);
        }
    }
    return out;
}

// One prefill from an empty cache, timed end to end. llama_decode can return
// before the GPU is finished, so wait for it explicitly.
double time_prefill(llama_context * ctx, const std::vector<llama_token> & toks) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_synchronize(ctx);
    const clk::time_point t0 = clk::now();
    decode_all(ctx, toks, "prefill");
    llama_synchronize(ctx);
    return ms_since(t0);
}

// Greedy decode from whatever the context holds now, timed per token.
double time_decode(llama_context * ctx, int n_vocab, int n) {
    const clk::time_point t0 = clk::now();
    for (int i = 0; i < n; i++) {
        llama_token t = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        decode_one(ctx, t, "decode");
    }
    llama_synchronize(ctx);
    return ms_since(t0);
}

std::vector<int> parse_lengths(const std::string & csv) {
    std::vector<int> out;
    size_t i = 0;
    while (i < csv.size()) {
        size_t j = csv.find(',', i);
        if (j == std::string::npos) j = csv.size();
        int n = atoi(csv.substr(i, j - i).c_str());
        if (n > 1) out.push_back(n);
        i = j + 1;
    }
    if (out.empty()) die("--lens needs at least one length above 1");
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The three laws of robotics, as written by Asimov, are:";
    std::string lens   = "512,2048";
    int  n_ctx    = 0;      // 0 = size it from the longest prompt
    int  n_ubatch = 512;    // llama.cpp's own default, and the chunk the GPU really works in
    int  warmup   = 8;
    int  iters    = 20;
    int  n_decode = 32;
    bool flash    = true;
    bool show_all = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--prompt"   && i + 1 < argc) prompt   = argv[++i];
        else if (a == "--lens"     && i + 1 < argc) lens     = argv[++i];
        else if (a == "--ctx"      && i + 1 < argc) n_ctx    = atoi(argv[++i]);
        else if (a == "--ubatch"   && i + 1 < argc) n_ubatch = atoi(argv[++i]);
        else if (a == "--warmup"   && i + 1 < argc) warmup   = atoi(argv[++i]);
        else if (a == "--iters"    && i + 1 < argc) iters    = atoi(argv[++i]);
        else if (a == "--decode"   && i + 1 < argc) n_decode = atoi(argv[++i]);
        else if (a == "--no-flash")                 flash    = false;
        else if (a == "--samples")                  show_all = true;
        else if (model_path.empty())                model_path = a;
        else die("unexpected argument " + a);
    }
    if (model_path.empty()) {
        die("usage: gpu-prefill <model.gguf> [--lens 512,2048] [--ubatch 512] "
            "[--warmup 8] [--iters 20] [--decode 32] [--ctx n] [--no-flash] [--samples] [--prompt text]");
    }
    if (iters < 1) die("--iters must be at least 1");

    const std::vector<int> lengths = parse_lengths(lens);
    const int longest = *std::max_element(lengths.begin(), lengths.end());
    if (n_ctx == 0) n_ctx = longest + n_decode + 64;
    if (n_ctx < longest + n_decode) die("--ctx is too small for the longest prompt plus its decode");

    ggml_backend_load_all();
    llama_backend_init();
    quiet_llama_logs();

    ggml_backend_dev_t gpu = find_gpu();
    if (!gpu) die("no GPU device registered");

    ggml_backend_dev_t gpus[] = { gpu, nullptr };
    llama_model_params mp = llama_model_default_params();
    mp.devices      = gpus;
    mp.n_gpu_layers = 999;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die("model load failed");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = std::max(n_ubatch, longest);
    cp.n_ubatch        = n_ubatch;
    cp.n_seq_max       = 1;
    cp.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_threads       = std::max(1u, std::thread::hardware_concurrency());
    cp.n_threads_batch = cp.n_threads;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context creation failed");

    printf("gpu:    %s (%s)\n", ggml_backend_dev_name(gpu), ggml_backend_dev_description(gpu));
    printf("model:  %s\n", model_path.c_str());
    printf("ctx %d, batch %d, ubatch %d, flash attention %s, %d warm-up + %d timed runs per length\n\n",
           n_ctx, cp.n_batch, cp.n_ubatch, flash ? "on" : "off", warmup, iters);

    struct row {
        int    len;
        stats  prefill;
        double first_ever;
        std::vector<double> discarded;
        std::vector<double> samples;
        double decode_tok_s;
    };
    std::vector<row> rows;
    bool first_prefill_of_process = true;

    for (int len : lengths) {
        const std::vector<llama_token> toks = prompt_of_length(vocab, prompt, len);

        row r;
        r.len        = len;
        r.first_ever = -1;

        for (int i = 0; i < warmup; i++) {
            const double ms = time_prefill(ctx, toks);
            r.discarded.push_back(ms);
            if (first_prefill_of_process) {
                r.first_ever = ms;
                first_prefill_of_process = false;
            }
        }

        std::vector<double> kept;
        kept.reserve(iters);
        for (int i = 0; i < iters; i++) kept.push_back(time_prefill(ctx, toks));
        r.prefill = summarise(kept);
        r.samples = kept;

        // The cache still holds the last prefill, so decode continues from it.
        if (n_decode > 0) {
            const double ms = time_decode(ctx, n_vocab, n_decode);
            r.decode_tok_s = n_decode * 1000.0 / ms;
        } else {
            r.decode_tok_s = 0;
        }

        printf("%5d tokens  warm-up: ", len);
        for (double d : r.discarded) printf("%.1f ", d);
        printf("ms   then %d runs: mean %.1f ms, median %.1f, %.1f-%.1f, sd %.1f (%.1f%%)\n",
               iters, r.prefill.mean, r.prefill.median, r.prefill.lo, r.prefill.hi,
               r.prefill.sd, r.prefill.mean > 0 ? 100.0 * r.prefill.sd / r.prefill.mean : 0.0);

        if (show_all) {
            printf("              every timed run: ");
            for (double d : r.samples) printf("%.0f ", d);
            printf("\n");
        }

        rows.push_back(r);
    }

    printf("\n%-8s %12s %12s %13s %13s\n", "tokens", "median ms", "mean ms", "prefill tok/s", "decode tok/s");
    for (const row & r : rows) {
        printf("%-8d %12.1f %12.1f %13.0f %13.1f\n",
               r.len, r.prefill.median, r.prefill.mean,
               r.len * 1000.0 / r.prefill.median, r.decode_tok_s);
    }

    // The point of the whole exercise: say out loud how far off the very first
    // prefill of the process was, so nobody quotes a cold number again.
    if (!rows.empty() && rows[0].first_ever > 0) {
        const double cold  = rows[0].first_ever;
        const double warm  = rows[0].prefill.median;
        const double ratio = cold / warm;
        printf("\nFirst prefill of this process: %.1f ms, against a warm median of %.1f ms at %d "
               "tokens (%.1fx).\n", cold, warm, rows[0].len, ratio);
        if (ratio > 2.0) {
            printf("A gap that size is the Vulkan shader pipelines being compiled. The driver caches "
                   "them, so the next launch will most likely not show it.\n");
        } else {
            printf("No compile cost this time; the driver had the pipelines cached. Which launch "
                   "pays it is not predictable, which is what the warm-up runs are for.\n");
        }
        printf("Either way, quote the median. Do not quote a prefill timed once.\n");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
