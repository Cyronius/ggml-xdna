# Zest model comparison: Lemonade on Strix Point

Measured on 2026-10-21, using the official Lemonade 2026.43.3 server and the
Windows preview's unmodified llama.cpp b10944 Vulkan backend plus `ggml-xdna.dll`.
These are actual Lemonade API requests, not standalone `llama-bench` results.

**This is Strix Point: Ryzen AI 9 HX PRO 370 / Radeon 890M.** The laptop has
96 GB of DDR5-5600 installed. Its 890M has 16 compute units; high-end Strix Halo's
Radeon 8060S has 40. Halo has a substantially larger GPU, but we haven't measured
Zest on it. A faster GPU baseline could reduce the NPU's relative benefit.
[AMD's GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html).

## Same prose, several models

Both throughput columns are medians of three runs. Speedup is the median of
the three **paired** GPU-time / NPU-time ratios, with the full range in brackets.
It is not the ratio of the two throughput medians. Tokenizers produce different
prompt lengths from the same prose.

| Model / GGUF | Prompt tokens | GPU only (tok/s) | NPU + GPU (tok/s) | Paired speedup (range) | Answer checks, GPU / NPU |
|---|---:|---:|---:|---:|---|
| Qwen3.8-27B UD-IQ3_S | 2,769 | 50 | 74 | **1.54×** (1.47–1.55) | 3/3 · 3/3 |
| LFM2.5-2.6B Q8_0 | 2,823 | 772 | 1,190 | **1.54×** (1.11–1.57) | 3/3 · 3/3 |
| Qwen3-4B Q4_K_M | 2,785 | 415 | 597 | **1.46×** (1.44–1.60) | 3/3 · 3/3 |
| LFM2.5-1.2B-Instruct Q8_0 | 2,846 | 1,769 | 2,600 | **1.46×** (1.44–1.48) | 2/3 · 2/3 |
| LFM2.5-350M Q8_0 | 2,846 | 5,338 | 5,897 | **1.14×** (1.05–1.15) | 2/3 · 2/3 |
| Bonsai 2 2B Q2_0 | — | — | — | Unsupported: garbled output | 0/3 · 0/3 |


The answer checks are two planted-code retrieval questions and one arithmetic
question, run separately in each mode. They are a small sanity check, not a
model-quality benchmark or a replacement for distribution-level accuracy tests.
Qwen3-4B answered the arithmetic correctly but returned an equation instead of
only the integer in both modes. LFM2.5-1.2B and 350M missed that question in both
modes. The complete outputs and formatting scores are in the JSON evidence.

LFM2.5-350M has the largest raw throughput here, but it is a much smaller model
with a modest gain in this sitting. The backend still warns that narrow models
have been slower with the NPU in earlier tests. Prompt shape and runtime state
matter even for the same GGUF.

## Qwen3.8-27B and chunk alignment

The main comparison's 2,769-token prompt contains a 2,048-token chunk and a
721-token tail. The default policy leaves that small tail on the GPU. Its
NPU weight copies also hit the default 20 GB cap: the trace reports the first
49 layers on the NPU and the rest on the GPU.

A separate exact **2,048-token** prose request measured **1.90×** median paired speedup (1.75–2.28×), with median throughput of **49.8 GPU-only vs 91.5 NPU + GPU tokens/s**. This is three additional warm pairs, using the same runtime, batch sizes, and default memory cap. The fixture was repeated and truncated to align one complete prompt chunk; it is a different workload from the table above.

To repeat it, use the same command with `-Copies 1 -Text docs/results/headline-models-2026-10-21/aligned-prose.txt` and a fresh output directory. [Raw aligned responses](results/headline-models-2026-10-21/qwen38-27b-aligned.json) · [Aligned NPU trace](results/headline-models-2026-10-21/qwen38-27b-aligned-trace.txt).

## Bonsai 2

The official **Ternary-Bonsai-2-2B-Q2_0.gguf** loads in the bundled runtime, but produced garbled text and scored **0/3 correct answers in GPU mode and 0/3 in NPU mode**. We did not run or publish a timed benchmark for this unsupported configuration. [Complete failed checks](results/headline-models-2026-10-21/bonsai2-2b-unsupported.json) retain the actual output. Since the GPU baseline also fails, this is not evidence of an NPU-only regression.

PrismML's [format notes](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/MODEL-FORMATS.md#bonsai-2-formats)
say Bonsai 2 needs its Hadamard/sign-flip runtime integration as well as Q2_0
decoding. Some older runtimes can load the format yet produce gibberish. This
preview is pinned to b10944, so neither a successful load nor a token counter
alone establishes useful Bonsai 2 support. PQ2_0 and PTQ1_0 additionally require
the PrismML fork. Replacing the backend with that fork requires a separate port
and validation of the add-on.

## How to repeat it

Start an isolated server with [the Lemonade launcher](lemonade.md). Use the current
repository's measurement scripts with the r2 preview's pinned binaries; the r2 ZIP
predates the helper's `-MaxTokens` and settings-receipt updates. Then run:

```powershell
tools/bench-lemonade.ps1 `
    -Model extra.Qwen3.8-27B-UD-IQ3_S `
    -BaseUrl http://127.0.0.1:13325/api/v1 `
    -Rounds 3 -Copies 8 -MaxTokens 16 `
    -OutDir third_party/my-qwen27-results
```

For each pair, the script unloads and loads the model in each mode, finishes one
full uncached warmup, then measures another uncached request. Pair order alternates
GPU/NPU, NPU/GPU, GPU/NPU. Both use context 8,192, batch and micro-batch 2,048,
flash attention, one slot, temperature zero, and at most 16 generated tokens.
The script rejects cached prompts, unequal paired token counts, or prompts under
2,048 tokens. The fixture is eight copies of `tests/prompts/prose.txt` plus a
summary request.

`enable_thinking=false` is requested, as recorded by `thinking: false` in the
saved settings. LFM2.5-2.6B still emits reasoning. The 16-token speed request can
end before an answer appears; it measures prefill, not answer completion. Separate
answer checks allow 256 output tokens (32 for the initial Qwen3-4B checks) and
retain reasoning, finish reason, answer correctness, and formatting results.

Times come from the backend's `timings.prompt_ms`. They exclude model load and
warmup and are **not client streaming time to first visible answer or total chat
latency**. These comparisons use the bundled b10944 Vulkan GPU baseline, not
Lemonade's default newer backends. Reply generation stays on the GPU; short decode
samples here don't establish a decode speedup.

This was one sitting on one laptop. Background load, temperature, and package
power were not controlled. The NPU's weight-copy budget was left at its
default; no failure injection or host-only emulation was enabled.

Positive hardware NPU-call counts are retained for every timed model. The tiny
350M model's two-request benchmark loads don't reach the trace's 196-matmul print
threshold; its positive call count comes from the subsequent answer-check load
using the same devices and arguments.

## Files and evidence

The downloaded GGUFs were checked against Hugging Face's LFS SHA-256 values.
Pinned revisions, model hashes, binary hashes, raw responses, timings, settings,
answer checks, and trace excerpts are in
[the evidence directory](results/headline-models-2026-10-21/).

- [CSV summary](results/headline-models-2026-10-21/summary.csv)
- [Runtime and source receipt](results/headline-models-2026-10-21/environment.json)
- [Qwen3.8-27B GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- [LFM2.5-2.6B GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF)
- [LFM2.5-1.2B GGUF](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF)
- [LFM2.5-350M GGUF](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF)
- [Bonsai 2 2B GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf)

The earlier [standalone results](technical-guide.md#results) used a different
workload and date. Their larger Qwen speedup shouldn't be mixed into this table.
