// Stage 0 of the hybrid prefill plan (.claude/plans/hybrid-prefill-amd-design.md).
//
// Question: can a prompt prefilled in one llama.cpp context be handed to a
// different context on a different backend through the public sequence-state
// API, and can we take that state apart and rebuild it ourselves?
//
// No NPU here. The source context runs on the CPU, standing in for the NPU,
// and the target runs on the GPU, which is where decode will live. Three runs:
//
//   A  source: prefill on the CPU, export the state, keep decoding on the CPU
//   B  target: import the state into a GPU context, decode there
//   C  control: the GPU prefills and decodes on its own
//
// The state is exported from A, parsed into fields, written back by our own
// writer, and only the rewritten bytes go into B. So B passing means the
// writer produces what a context accepts, not just that the API round-trips.
//
// The handoff carries every prompt token but the last. B then decodes that
// last token itself, which is what gives it logits to sample from; a bare
// cache import has none. The real driver will do the same: the NPU prefills,
// the GPU runs the final token and gets the first prediction as a by-product.
//
// Traces: HYBRID-KV-STATE-FORMAT

#include "common.h"
#include "kv_state.h"

#include <thread>

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The three laws of robotics, as written by Asimov, are:";
    int n_decode = 24;
    int n_ctx    = 4096;
    int repeat   = 1;   // tile the prompt to get a longer one

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--prompt" && i + 1 < argc) prompt   = argv[++i];
        else if (a == "--n"      && i + 1 < argc) n_decode = atoi(argv[++i]);
        else if (a == "--ctx"    && i + 1 < argc) n_ctx    = atoi(argv[++i]);
        else if (a == "--repeat" && i + 1 < argc) repeat   = atoi(argv[++i]);
        else if (model_path.empty())              model_path = a;
        else die("unexpected argument " + a);
    }
    if (model_path.empty()) die("usage: kv-handoff <model.gguf> [--prompt text] [--n tokens] [--ctx n] [--repeat k]");
    {
        std::string base = prompt;
        for (int i = 1; i < repeat; i++) prompt += " " + base;
    }

    ggml_backend_load_all();
    llama_backend_init();
    quiet_llama_logs();

    ggml_backend_dev_t gpu = find_gpu();
    if (!gpu) die("no GPU device registered");
    printf("gpu: %s (%s)\n", ggml_backend_dev_name(gpu), ggml_backend_dev_description(gpu));

    // Two loads of the same file: one pinned to the CPU, one on the GPU. The
    // real driver will load once for the GPU and read the file separately for
    // the NPU; here the CPU copy stands in for that second reader.
    ggml_backend_dev_t none[] = { nullptr };
    ggml_backend_dev_t gpus[] = { gpu, nullptr };

    llama_model_params mp_cpu = llama_model_default_params();
    mp_cpu.devices      = none;
    mp_cpu.n_gpu_layers = 0;

    llama_model_params mp_gpu = llama_model_default_params();
    mp_gpu.devices      = gpus;
    mp_gpu.n_gpu_layers = 999;

    llama_model * m_cpu = llama_model_load_from_file(model_path.c_str(), mp_cpu);
    if (!m_cpu) die("load (cpu) failed");
    llama_model * m_gpu = llama_model_load_from_file(model_path.c_str(), mp_gpu);
    if (!m_gpu) die("load (gpu) failed");

    const llama_vocab * vocab = llama_model_get_vocab(m_cpu);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const std::vector<llama_token> prompt_toks = tokenize(vocab, prompt);
    const int n_prompt = (int) prompt_toks.size();
    if (n_prompt < 2) die("prompt too short");

    // Flash attention on for both, so V is stored row-per-cell on both sides
    // and the import's layout check passes. That is also the layout an NPU
    // kernel would naturally write.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = std::max(512, n_prompt);
    cp.n_ubatch        = cp.n_batch;
    cp.n_seq_max       = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = std::max(1u, std::thread::hardware_concurrency());
    cp.n_threads_batch = cp.n_threads;

    printf("model: %s\n", model_path.c_str());
    printf("prompt: %d tokens, decode: %d tokens, layers: %d, kv heads: %d\n\n",
           n_prompt, n_decode, llama_model_n_layer(m_cpu), llama_model_n_head_kv(m_cpu));

    // ---- A: source, on the CPU ------------------------------------------------
    run A;
    llama_context * ctx_a = llama_init_from_model(m_cpu, cp);
    if (!ctx_a) die("context A");
    {
        clk::time_point t0 = clk::now();
        decode_all(ctx_a, prompt_toks, "prefill A");
        A.prefill_ms = ms_since(t0);
    }

    const size_t st_size = llama_state_seq_get_size(ctx_a, 0);
    std::vector<uint8_t> blob(st_size);
    if (llama_state_seq_get_data(ctx_a, blob.data(), blob.size(), 0) != st_size) die("export A");

    A.first_logits = take_logits(ctx_a, n_vocab);
    greedy(ctx_a, n_vocab, n_decode, A);

    // ---- take it apart and rebuild it -----------------------------------------
    kv_state st;
    std::string err;
    if (!kv_state_parse(blob.data(), blob.size(), st, err)) die("parse: " + err);
    const std::vector<uint8_t> rebuilt = kv_state_write(st);
    const bool identical = rebuilt == blob;

    printf("state: %zu bytes, %u cells, %zu layers, v_trans=%d, k row %llu B (type %d)\n",
           blob.size(), st.cell_count, st.n_layer(), (int) st.v_trans,
           (unsigned long long) st.k[0].row_bytes, st.k[0].type);
    printf("       positions %d..%d, rebuilt by our writer: %s\n\n",
           st.pos.front(), st.pos.back(), identical ? "byte-identical" : "DIFFERS");

    // ---- B: target, on the GPU, from the rebuilt bytes -------------------------
    run B;
    llama_context * ctx_b = llama_init_from_model(m_gpu, cp);
    if (!ctx_b) die("context B");
    size_t n_set = 0;
    {
        clk::time_point t0 = clk::now();
        n_set = llama_state_seq_set_data(ctx_b, rebuilt.data(), rebuilt.size(), 0);
        B.import_ms = ms_since(t0);
    }
    if (n_set == 0) die("import into B was refused");

    // Drop the last prompt token's cell and run that token here to get logits.
    if (!llama_memory_seq_rm(llama_get_memory(ctx_b), 0, n_prompt - 1, -1)) die("seq_rm B");
    {
        clk::time_point t0 = clk::now();
        decode_one(ctx_b, prompt_toks[n_prompt - 1], "last prompt token on B");
        B.last_tok_ms = ms_since(t0);
    }
    B.prefill_ms = B.import_ms + B.last_tok_ms;
    B.first_logits = take_logits(ctx_b, n_vocab);
    greedy(ctx_b, n_vocab, n_decode, B);

    // Same import again into the now-warm context, to tell a first-use cost
    // from a per-import one.
    {
        llama_memory_clear(llama_get_memory(ctx_b), true);
        clk::time_point t0 = clk::now();
        if (llama_state_seq_set_data(ctx_b, rebuilt.data(), rebuilt.size(), 0) == 0) die("second import into B");
        B.import2_ms = ms_since(t0);
    }

    // ---- C: control, GPU alone -------------------------------------------------
    run C;
    llama_context * ctx_c = llama_init_from_model(m_gpu, cp);
    if (!ctx_c) die("context C");
    {
        clk::time_point t0 = clk::now();
        decode_all(ctx_c, prompt_toks, "prefill C");
        C.prefill_ms = ms_since(t0);
    }
    C.first_logits = take_logits(ctx_c, n_vocab);
    greedy(ctx_c, n_vocab, n_decode, C);

    // ---- report ----------------------------------------------------------------
    print_run("A cpu source", A, n_vocab, n_decode);
    print_run("B gpu target", B, n_vocab, n_decode);
    print_run("C gpu control", C, n_vocab, n_decode);
    printf("\nB's prefill column is import %.1f ms + last-token decode %.1f ms; "
           "a second import into the same warm context took %.1f ms\n\n",
           B.import_ms, B.last_tok_ms, B.import2_ms);

    const bool first_ba = argmax(B.first_logits.data(), n_vocab) == argmax(A.first_logits.data(), n_vocab);
    const bool first_bc = argmax(B.first_logits.data(), n_vocab) == argmax(C.first_logits.data(), n_vocab);
    const double cos_ba = cosine(B.first_logits, A.first_logits);
    const double cos_bc = cosine(B.first_logits, C.first_logits);
    const double cos_ac = cosine(A.first_logits, C.first_logits);

    printf("first-token logits  cos(B,A) %.6f  cos(B,C) %.6f  cos(A,C) %.6f\n", cos_ba, cos_bc, cos_ac);
    printf("greedy continuation matches  B vs A: %d/%d   B vs C: %d/%d   A vs C: %d/%d\n",
           match_len(B.toks, A.toks), n_decode, match_len(B.toks, C.toks), n_decode,
           match_len(A.toks, C.toks), n_decode);

    printf("\nB says: ");
    print_pieces(vocab, B.toks);
    printf("\n\n");

    checker check;
    check(identical,       "state parsed and rewritten byte-identically");
    check(n_set > 0,       "rewritten state accepted by the GPU context");
    check(first_ba,        "GPU target predicts the same first token as the CPU source");
    check(cos_ba >= 0.999, "first-token logits agree with the source (cosine >= 0.999)");
    check(first_bc,        "GPU target predicts the same first token as the GPU control");

    llama_free(ctx_a);
    llama_free(ctx_b);
    llama_free(ctx_c);
    llama_model_free(m_cpu);
    llama_model_free(m_gpu);
    llama_backend_free();
    return check.finish();
}
