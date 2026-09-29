// The second half of stage 0 (.claude/plans/hybrid-prefill-amd-design.md):
// keys and values computed by our own code, not copied out of llama.cpp,
// handed to a GPU context and decoded.
//
//   A  llama.cpp on the CPU prefills; its exported state is the yardstick
//   R  our host reference (hybrid/qwen3_ref.cpp) prefills the same tokens
//      from the same file; its keys and values are compared to A's layer by
//      layer, and its last-token logits to A's
//   B  a GPU context is given a state assembled from R's keys and values,
//      runs the last prompt token, and decodes
//   C  the GPU on its own, as the control
//
// R stands in for the NPU. When the kernel exists it replaces R's layer
// function and nothing else here changes.
//
// Two yardsticks, and the model file decides which this is. On an
// unquantised file (F32, F16, BF16) llama.cpp computes in full precision and
// the layer-by-layer comparison is a real check on R, so it is a gate. On a
// quantised file llama.cpp rounds its activations to 8-bit blocks before every
// matmul, which throws its keys and values off by up to 100% on a few outlier
// tokens; the layer table is then printed for information and only the
// end-to-end result is gated. Run both: the first proves R, the second proves
// the handoff on the format the NPU will actually use.
//
// Traces: HYBRID-REF-PREFILL

#include "common.h"
#include "kv_state.h"
#include "qwen3_ref.h"

#include "ggml.h"

#include <cstring>
#include <thread>

// Relative RMS error of each row (one prompt position) between two
// half-float row blocks.
static std::vector<double> row_errors(const uint16_t * a, const uint16_t * b, size_t n_rows, size_t row_el) {
    std::vector<double> e(n_rows);
    std::vector<float> fa(row_el), fb(row_el);
    for (size_t r = 0; r < n_rows; r++) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) a + r * row_el, fa.data(), row_el);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) b + r * row_el, fb.data(), row_el);
        double se = 0, sb = 0;
        for (size_t i = 0; i < row_el; i++) {
            const double d = fa[i] - fb[i];
            se += d * d; sb += (double) fb[i] * fb[i];
        }
        e[r] = std::sqrt(se / std::max(sb, 1e-300));
    }
    return e;
}

// Relative RMS error and cosine between two half-float rows
static void compare_rows(const uint16_t * a, const uint16_t * b, size_t n, double & nrmse, double & cos) {
    std::vector<float> fa(n), fb(n);
    ggml_fp16_to_fp32_row((const ggml_fp16_t *) a, fa.data(), n);
    ggml_fp16_to_fp32_row((const ggml_fp16_t *) b, fb.data(), n);
    double se = 0, sb = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = fa[i] - fb[i];
        se += d * d; sb += (double) fb[i] * fb[i];
        dot += (double) fa[i] * fb[i]; na += (double) fa[i] * fa[i]; nb += (double) fb[i] * fb[i];
    }
    nrmse = std::sqrt(se / std::max(sb, 1e-300));
    cos   = dot / std::sqrt(na * nb);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The three laws of robotics, as written by Asimov, are:";
    int n_decode = 24, n_ctx = 4096, repeat = 1;
    int n_threads = std::max(1u, std::thread::hardware_concurrency());

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--prompt"  && i + 1 < argc) prompt    = argv[++i];
        else if (a == "--n"       && i + 1 < argc) n_decode  = atoi(argv[++i]);
        else if (a == "--ctx"     && i + 1 < argc) n_ctx     = atoi(argv[++i]);
        else if (a == "--repeat"  && i + 1 < argc) repeat    = atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (model_path.empty())               model_path = a;
        else die("unexpected argument " + a);
    }
    if (model_path.empty()) die("usage: ref-handoff <qwen3.gguf> [--prompt text] [--n tokens] [--repeat k] [--threads n]");
    {
        std::string base = prompt;
        for (int i = 1; i < repeat; i++) prompt += " " + base;
    }

    ggml_backend_load_all();
    llama_backend_init();
    quiet_llama_logs();

    ggml_backend_dev_t gpu = find_gpu();
    if (!gpu) die("no GPU device registered");

    ggml_backend_dev_t none[] = { nullptr };
    ggml_backend_dev_t gpus[] = { gpu, nullptr };
    llama_model_params mp_cpu = llama_model_default_params();
    mp_cpu.devices = none;  mp_cpu.n_gpu_layers = 0;
    llama_model_params mp_gpu = llama_model_default_params();
    mp_gpu.devices = gpus;  mp_gpu.n_gpu_layers = 999;

    llama_model * m_cpu = llama_model_load_from_file(model_path.c_str(), mp_cpu);
    if (!m_cpu) die("load (cpu) failed");
    llama_model * m_gpu = llama_model_load_from_file(model_path.c_str(), mp_gpu);
    if (!m_gpu) die("load (gpu) failed");

    const llama_vocab * vocab = llama_model_get_vocab(m_cpu);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const std::vector<llama_token> toks = tokenize(vocab, prompt);
    const int n_prompt = (int) toks.size();
    if (n_prompt < 2) die("prompt too short");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = cp.n_ubatch = std::max(512, n_prompt);
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads = cp.n_threads_batch = n_threads;

    printf("model: %s\nprompt: %d tokens, decode: %d, threads: %d\n\n", model_path.c_str(), n_prompt, n_decode, n_threads);

    // ---- A: llama.cpp on the CPU, the yardstick -------------------------------
    run A;
    llama_context * ctx_a = llama_init_from_model(m_cpu, cp);
    if (!ctx_a) die("context A");
    {
        clk::time_point t0 = clk::now();
        decode_all(ctx_a, toks, "prefill A");
        A.prefill_ms = ms_since(t0);
    }
    std::vector<uint8_t> blob(llama_state_seq_get_size(ctx_a, 0));
    if (llama_state_seq_get_data(ctx_a, blob.data(), blob.size(), 0) != blob.size()) die("export A");
    A.first_logits = take_logits(ctx_a, n_vocab);
    greedy(ctx_a, n_vocab, n_decode, A);

    kv_state ref;
    std::string err;
    if (!kv_state_parse(blob.data(), blob.size(), ref, err)) die("parse A: " + err);
    if (ref.v_trans) die("A's cache is transposed; expected flash attention on");

    // ---- R: our reference ----------------------------------------------------
    qwen3_ref model;
    if (!model.load(model_path, err)) die("reference load: " + err);
    const qwen3_hparams & hp = model.hp();
    if ((size_t) hp.n_layer != ref.n_layer()) die("layer count differs from llama.cpp's cache");
    const size_t row_el = (size_t) hp.n_head_kv * hp.head_dim;
    if (ref.k[0].row_bytes != row_el * 2 || ref.k[0].type != GGML_TYPE_F16) die("cache row is not f16 of the expected width");

    qwen3_prefill_out R;
    model.prefill(std::vector<int32_t>(toks.begin(), toks.end()), R, n_threads);
    printf("reference prefill: %.0f ms on %d threads\n", R.ms, n_threads);
    printf("weights are %s, so the layer-by-layer comparison is %s\n\n",
           model.quantized() ? "quantised" : "full precision",
           model.quantized() ? "informational (llama.cpp rounds activations to 8-bit)" : "a gate");

    // ---- compare R against A, layer by layer ---------------------------------
    // Position 0 only ever attends to itself, so its keys and values cannot
    // depend on how long the prompt is. Its error column is the same
    // quantity at every prompt length; the whole-layer columns are not.
    printf("layer   K nrmse   K cos       V nrmse   V cos      K@pos0    V@pos0\n");
    double worst_k = 0, worst_v = 0, worst_cos = 1;
    int worst_layer = 0;
    for (int il = 0; il < hp.n_layer; il++) {
        double nk, ck, nv, cv;
        compare_rows(R.k[il].data(), (const uint16_t *) ref.k[il].data.data(), R.k[il].size(), nk, ck);
        compare_rows(R.v[il].data(), (const uint16_t *) ref.v[il].data.data(), R.v[il].size(), nv, cv);
        const std::vector<double> ek = row_errors(R.k[il].data(), (const uint16_t *) ref.k[il].data.data(), n_prompt, row_el);
        const std::vector<double> ev = row_errors(R.v[il].data(), (const uint16_t *) ref.v[il].data.data(), n_prompt, row_el);
        if (nk > worst_k) { worst_k = nk; worst_layer = il; }
        worst_v = std::max(worst_v, nv);
        worst_cos = std::min({ worst_cos, ck, cv });
        if (il < 3 || il % 5 == 4 || il == hp.n_layer - 1) {
            printf("%5d   %.2e  %.6f   %.2e  %.6f   %.2e  %.2e\n", il, nk, ck, nv, cv, ek[0], ev[0]);
        }
    }
    printf("worst   K nrmse %.2e   V nrmse %.2e   cos %.6f  (layer %d)\n", worst_k, worst_v, worst_cos, worst_layer);

    // Where in the prompt the worst layer's error sits: spread evenly across
    // positions, or concentrated on a few.
    {
        std::vector<double> ek = row_errors(R.k[worst_layer].data(), (const uint16_t *) ref.k[worst_layer].data.data(), n_prompt, row_el);
        std::vector<double> sorted = ek;
        std::sort(sorted.begin(), sorted.end());
        const size_t at = std::max_element(ek.begin(), ek.end()) - ek.begin();
        std::vector<size_t> over;
        for (size_t p = 0; p < ek.size(); p++) if (ek[p] > 0.05) over.push_back(p);
        char piece[64] = { 0 };
        llama_token_to_piece(vocab, toks[at], piece, sizeof(piece) - 1, 0, true);
        printf("layer %d per-position K error: min %.2e  median %.2e  p90 %.2e  max %.2e at position %zu (token %d '%s'); "
               "%zu of %d positions above 0.05\n",
               worst_layer, sorted.front(), sorted[n_prompt / 2], sorted[(size_t) (n_prompt * 0.9)], sorted.back(),
               at, toks[at], piece, over.size(), n_prompt);
        if (!over.empty()) {
            printf("  those positions (token id):");
            for (size_t i = 0; i < over.size() && i < 16; i++) printf(" %zu(%d)", over[i], toks[over[i]]);
            if (over.size() > 16) printf(" ...");
            printf("\n");
        }
        printf("\n");
    }

    const llama_token first_r = argmax(R.logits_last.data(), n_vocab);
    const llama_token first_a = argmax(A.first_logits.data(), n_vocab);
    const double cos_ra = cosine(R.logits_last, A.first_logits);
    printf("last-token logits: reference argmax %d, llama.cpp argmax %d, cosine %.6f\n\n", first_r, first_a, cos_ra);

    // ---- B: a GPU context fed our keys and values ----------------------------
    kv_state ours;
    ours.seq_id = 0;
    ours.cell_count = n_prompt;
    ours.pos.resize(n_prompt);
    ours.seq_ids.assign(n_prompt, { 0 });
    for (int i = 0; i < n_prompt; i++) ours.pos[i] = i;
    ours.v_trans = false;
    ours.k.resize(hp.n_layer);
    ours.v.resize(hp.n_layer);
    for (int il = 0; il < hp.n_layer; il++) {
        auto fill = [&](kv_state::layer_rows & L, const std::vector<uint16_t> & src) {
            L.type = GGML_TYPE_F16;
            L.row_bytes = row_el * 2;
            L.data.resize(src.size() * 2);
            memcpy(L.data.data(), src.data(), L.data.size());
        };
        fill(ours.k[il], R.k[il]);
        fill(ours.v[il], R.v[il]);
    }
    const std::vector<uint8_t> our_blob = kv_state_write(ours);
    const bool same_size = our_blob.size() == blob.size();

    run B;
    llama_context * ctx_b = llama_init_from_model(m_gpu, cp);
    if (!ctx_b) die("context B");
    size_t n_set;
    {
        clk::time_point t0 = clk::now();
        n_set = llama_state_seq_set_data(ctx_b, our_blob.data(), our_blob.size(), 0);
        B.import_ms = ms_since(t0);
    }
    if (n_set == 0) die("import of our state into B was refused");
    if (!llama_memory_seq_rm(llama_get_memory(ctx_b), 0, n_prompt - 1, -1)) die("seq_rm B");
    {
        clk::time_point t0 = clk::now();
        decode_one(ctx_b, toks[n_prompt - 1], "last prompt token on B");
        B.last_tok_ms = ms_since(t0);
    }
    B.prefill_ms = B.import_ms + B.last_tok_ms;
    B.first_logits = take_logits(ctx_b, n_vocab);
    greedy(ctx_b, n_vocab, n_decode, B);

    // ---- C: control -----------------------------------------------------------
    run C;
    llama_context * ctx_c = llama_init_from_model(m_gpu, cp);
    if (!ctx_c) die("context C");
    {
        clk::time_point t0 = clk::now();
        decode_all(ctx_c, toks, "prefill C");
        C.prefill_ms = ms_since(t0);
    }
    C.first_logits = take_logits(ctx_c, n_vocab);
    greedy(ctx_c, n_vocab, n_decode, C);

    // ---- report ----------------------------------------------------------------
    print_run("A cpu llama", A, n_vocab, n_decode);
    print_run("B gpu <- ours", B, n_vocab, n_decode);
    print_run("C gpu control", C, n_vocab, n_decode);
    const double cos_ba = cosine(B.first_logits, A.first_logits);
    const double cos_bc = cosine(B.first_logits, C.first_logits);
    printf("\nfirst-token logits  cos(B,A) %.6f  cos(B,C) %.6f\n", cos_ba, cos_bc);
    printf("greedy continuation matches  B vs A: %d/%d   B vs C: %d/%d\n",
           match_len(B.toks, A.toks), n_decode, match_len(B.toks, C.toks), n_decode);
    printf("\nB says: ");
    print_pieces(vocab, B.toks);
    printf("\n\n");

    checker check;
    check(same_size,                          "our state is the same size as llama.cpp's");
    if (model.quantized()) {
        printf("skip every layer's keys and values agree with llama.cpp (quantised weights: not a gate, worst cosine %.6f)\n", worst_cos);
    } else {
        check(worst_cos > 0.999,              "every layer's keys and values agree with llama.cpp (cosine > 0.999)");
    }
    check(first_r == first_a,                 "reference's last-token prediction matches llama.cpp's");
    check(n_set > 0,                          "our state accepted by the GPU context");
    check(argmax(B.first_logits.data(), n_vocab) == first_a, "GPU fed our state predicts the same first token as llama.cpp");
    check(cos_ba >= 0.99,                     "first-token logits agree with llama.cpp (cosine >= 0.99)");

    llama_free(ctx_a); llama_free(ctx_b); llama_free(ctx_c);
    llama_model_free(m_cpu); llama_model_free(m_gpu);
    llama_backend_free();
    return check.finish();
}
