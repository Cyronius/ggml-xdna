// Small helpers shared by the hybrid/ programs: driving a llama.cpp context
// from the C API, timing, and comparing runs.
#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

inline double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

[[noreturn]] inline void die(const std::string & what) {
    fprintf(stderr, "error: %s\n", what.c_str());
    exit(1);
}

// Only warnings and errors from llama.cpp; its progress dots and load logs
// drown the numbers otherwise.
inline void quiet_llama_logs() {
    llama_log_set([](ggml_log_level lvl, const char * text, void *) {
        if (lvl >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
    }, nullptr);
}

inline ggml_backend_dev_t find_gpu() {
    if (auto d = ggml_backend_dev_by_name("Vulkan0")) return d;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) return d;
    }
    return nullptr;
}

inline std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, true, true);
    std::vector<llama_token> out(n);
    n = llama_tokenize(vocab, text.c_str(), (int) text.size(), out.data(), n, true, true);
    if (n < 0) die("tokenize failed");
    out.resize(n);
    return out;
}

inline llama_token argmax(const float * v, int n) {
    return (llama_token) (std::max_element(v, v + n) - v);
}

inline double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += (double) a[i] * b[i];
        na  += (double) a[i] * a[i];
        nb  += (double) b[i] * b[i];
    }
    return dot / std::sqrt(na * nb);
}

inline std::vector<float> take_logits(llama_context * ctx, int n_vocab) {
    const float * p = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(p, p + n_vocab);
}

inline void decode_one(llama_context * ctx, llama_token t, const char * what) {
    llama_batch b = llama_batch_get_one(&t, 1);
    if (llama_decode(ctx, b) != 0) die(std::string("decode failed: ") + what);
}

// In pieces of at most the context's batch size, as llama-bench feeds a
// prompt longer than -b.
inline void decode_all(llama_context * ctx, const std::vector<llama_token> & toks, const char * what) {
    const size_t n_batch = llama_n_batch(ctx);
    for (size_t i = 0; i < toks.size(); i += n_batch) {
        llama_batch b = llama_batch_get_one((llama_token *) toks.data() + i, (int) std::min(n_batch, toks.size() - i));
        if (llama_decode(ctx, b) != 0) die(std::string("decode failed: ") + what);
    }
}

// One run of a context: what it predicted first, what it said after, and
// how long the two phases took.
struct run {
    std::vector<float>       first_logits;
    std::vector<llama_token> toks;
    double prefill_ms = 0;
    double decode_ms  = 0;
    // target-side detail: the import, the one-token decode after it, and a
    // second import into the same warm context
    double import_ms = 0, last_tok_ms = 0, import2_ms = 0;
};

// Greedy continuation from whatever logits the context holds now. n tokens
// come out of n - 1 decode calls; the first is sampled from existing logits.
inline void greedy(llama_context * ctx, int n_vocab, int n, run & r) {
    const clk::time_point t0 = clk::now();
    for (int i = 0; i < n; i++) {
        llama_token t = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        r.toks.push_back(t);
        if (i + 1 < n) decode_one(ctx, t, "continuation");
    }
    r.decode_ms = ms_since(t0);
}

inline int match_len(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    int n = 0;
    while (n < (int) std::min(a.size(), b.size()) && a[n] == b[n]) n++;
    return n;
}

inline void print_run(const char * name, const run & r, int n_vocab, int n_decode) {
    printf("%-14s first=%6d  prefill %8.1f ms  decode %7.1f ms (%.1f tok/s)  ",
           name, argmax(r.first_logits.data(), n_vocab), r.prefill_ms, r.decode_ms,
           r.decode_ms > 0 ? (n_decode - 1) * 1000.0 / r.decode_ms : 0.0);
    for (llama_token t : r.toks) printf("%d ", t);
    printf("\n");
}

inline void print_pieces(const llama_vocab * vocab, const std::vector<llama_token> & toks) {
    for (llama_token t : toks) {
        char buf[256];
        int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        if (n > 0) fwrite(buf, 1, n, stdout);
    }
}

// Tally of PASS/FAIL lines at the end of a program.
struct checker {
    int failures = 0;
    void operator()(bool ok, const char * what) {
        printf("%s %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) failures++;
    }
    int finish() {
        printf("%s\n", failures ? "FAILED" : "ALL PASS");
        return failures ? 1 : 0;
    }
};
