# Hybrid prefill — requirements

Canonical spec for reading prompts on the NPU: prefill on the NPU, decode on
the integrated GPU through a stock llama.cpp context, GGUF as the model
format. Home repo: `npu-prefill-engine`.

Plan: `.claude/plans/hybrid-prefill-amd-design.md`. This spec grows as the
stages in that plan land.

**These requirements are written against `hybrid/`, which drives llama.cpp as
a library. That is a prototype shape, not the destination.** The destination
is a ggml backend in this same repo that claims a whole transformer block —
see the plan's "Where this lands" section. The requirements below were written
to survive that move: they fix what the NPU must compute and what a llama.cpp
context must accept, not how the code is packaged. `HYBRID-KV-STATE-FORMAT` is
the exception and becomes unnecessary once a backend writes into ggml's own
cache tensors.

The backend spec under `specs/xdna-backend/` covers the existing matmul-level
backend and is unchanged by any of this.

Tests: `specs/hybrid-prefill/tests/` is empty by design, for the same reason
as the backend spec. The verification binaries are built by the top-level
`CMakeLists.txt` from `hybrid/` because they link the pinned llama.cpp release.
They need a GGUF model file and the GPU, so run them from the release directory:

```
build.cmd
third_party\llama-b10944\kv-handoff.exe <model.gguf> [--repeat 34]
```

`gpu-prefill.exe` is built the same way and is a measurement rather than a
test: it has no pass or fail, it prints numbers, and HYBRID-PREFILL-BASELINE
says how to read them.

---

### HYBRID-KV-STATE-FORMAT: A prefilled prompt can be handed to a llama.cpp context
**Applies to:** npu-prefill-engine
**Verification:** test

The driver shall produce a per-sequence cache state that a llama.cpp context
of the pinned release accepts through `llama_state_seq_set_data`, such that
the context then decodes as if it had prefilled the prompt itself.

The state format is llama.cpp's own and is not under our control. What this
requirement fixes is that our writer reproduces it exactly: taking a state the
context exported apart into fields and writing it back must give the same
bytes, and a state assembled from fields must be accepted and decode
correctly. Anything else means the NPU's keys and values will land in the
wrong place silently.

Verified by `hybrid/kv-handoff.cpp`, which prefills on the CPU standing in for
the NPU, exports, parses, rewrites, imports the rewritten bytes into a GPU
context, decodes, and compares against the CPU continuing on its own and
against the GPU prefilling on its own.

**Acceptance criteria:**
- parse then write of an exported state is byte-identical to the export
- the rewritten state is accepted by a GPU context (`llama_state_seq_set_data`
  returns non-zero)
- the GPU context's first predicted token equals the CPU source's
- first-token logits have cosine similarity at least 0.999 with the source
- the GPU context's first predicted token equals the all-GPU control's

**Measured 2026-09-18** on Qwen3-1.7B Q4_0 (28 layers, 8 KV heads), flash
attention on, this box: all five hold at 15 and at 510 prompt tokens; the full
24-token greedy continuation matched across all three runs at both lengths;
import cost 0.2 ms at 15 tokens and 2.3 ms at 510 (58 MB of state). Also holds
on SmolLM2-135M, where the continuation diverged at token 7 on a near-tie,
which is why the criteria are on the first token and the logits rather than
the whole continuation.

---

### HYBRID-REF-PREFILL: The driver's own prefill produces the state llama.cpp would
**Applies to:** npu-prefill-engine
**Verification:** test

The driver shall compute every layer's keys and values for a prompt from the
GGUF file itself, without llama.cpp's help, such that a llama.cpp context
given that state decodes as if it had prefilled the prompt. This is the
contract the NPU kernel has to meet; until it exists, the host reference in
`hybrid/qwen3_ref.cpp` meets it and stands in.

The reference is the kernel's oracle. Its own correctness is established
against llama.cpp on an unquantised copy of the model, where llama.cpp
computes in full precision. On a quantised copy llama.cpp rounds activations
to 8-bit blocks of 32 before every matmul, which on a few outlier tokens moves
its keys by up to 100%; measured against that, per-layer agreement is
informational and only the end-to-end result counts.

Verified by `hybrid/ref-handoff.cpp`, run on both an F32 and a Q4_0 copy of
the same model. It gates layer-by-layer agreement only when the file is
unquantised, and gates the end-to-end result always.

**Acceptance criteria:**
- on the F32 file: every layer's keys and values have cosine similarity above
  0.999 with llama.cpp's exported state, and position 0's rows (which under
  causal attention cannot depend on prompt length) agree to within 1e-3
- on every file: the reference's last-token prediction equals llama.cpp's; a
  GPU context given the reference's state accepts it, predicts the same first
  token as llama.cpp, with first-token logit cosine at least 0.99

**Measured 2026-09-18** on Qwen3-1.7B, 510-token prompt, this box:

| file | worst layer cosine | worst K rel. error | position 0 K error | last-token logit cosine | 24-token continuation |
|---|---|---|---|---|---|
| F32 | 0.999963 | 4.7e-3 | 1e-5 to 6e-5 | 1.000000 | identical |
| Q4_0 | 0.977 (not gated) | 1.2e-1 | 2e-3 to 7e-3 | 0.999617 | identical |

On the Q4_0 file the disagreement sits on the same few tokens in every
repetition of the prompt (the pieces of "Asimov", " as", " robotics"), 43 of
510 positions, while the median position agrees to 3%. On the F32 file the two
touchiest positions are both the 'im' piece at 7%, everything else under 1e-3.

---

### HYBRID-Q4-PACK: GGUF 4-bit weights reach the NPU without being requantised
**Applies to:** npu-prefill-engine
**Verification:** test

The driver shall rearrange a GGUF `Q4_0` or `Q4_1` weight tensor into the
5120-byte chunk layout the open NPU kernels stream, changing nothing but the
arrangement and the width of the scale and minimum.

Two things have to hold. The chunks must dequantise to exactly the original
weights with the scale and minimum narrowed to bfloat16, which is all a chunk
can hold, so the narrowing is the only loss. And the bytes must be identical
to what openflowlm-next's own packer produces, because that is what the kernel
was built against.

`Q4_0` has no stored minimum and subtracts 8 from every code instead. That is
the same rule with a minimum of -8 times the scale, and it stays exact through
the narrowing because scaling by a power of two leaves the mantissa alone. So
neither type may lose anything beyond the narrowing.

Verified by `hybrid/test-q4-pack.cpp` against ggml's own dequantisation, and by
`tools/check-q4-pack.py` against openflowlm-next's `open_kernels/q4_1_pack.py`,
which it imports read-only.

**Acceptance criteria:**
- the packed pool dequantises bit-for-bit to the narrowed weights, for both types
- the packed bytes are identical to the reference packer's, for both types and
  for both band row splits (2 and 4)
- the narrowing costs under 5e-3 relative

**Measured 2026-09-18** on Qwen3-1.7B Q4_0, this box, at both row splits:

| tensor | type | shape | chunks | narrowing cost | vs reference packer |
|---|---|---|---|---|---|
| `blk.0.attn_k.weight` | Q4_0 | K 2048, 1024 rows | 256 | 1.8e-3 | 1310720 bytes identical |
| `blk.0.ffn_down.weight` | Q4_1 | K 6144, 2048 rows | 1536 | 3.7e-3 | 7864320 bytes identical |

---

### HYBRID-NPU-GEMM: Packed GGUF weights compute correctly on the NPU
**Applies to:** npu-prefill-engine
**Verification:** test

The driver shall drive one of openflowlm-next's built prefill GEMM kernels
with weights taken from a GGUF and packed by `q4_pack`, and activations
arranged by our own tiler, and get back the right answer.

This is the whole weight path on real hardware: read, pack, tile, dispatch,
compare. It is what proves the packer against the kernel rather than against
another packer, and it is the piece the fused block kernel will later slot
into unchanged.

The reference is the same multiply in double precision using the weights as
the chunks hold them, so what is measured is the kernel's own error and not
the cost of narrowing the scale. That cost is priced by HYBRID-Q4-PACK.

The built kernel fixes the shape, so the weight matrix is filled with however
many real tensors of the right row length are needed. Which tensors they are
does not matter; that they are real 4-bit GGUF weights packed by our own code
does.

Verified by `hybrid/test-npu-gemm.cpp`, which needs the NPU driver and a built
kernel directory and is skipped by the build when the driver is absent.

**Acceptance criteria:**
- every output value is finite
- relative Frobenius error at most 5e-3, which is the kernel's own gate
- every token's output has cosine similarity above 0.9999 with the reference

**Measured 2026-09-18** on Qwen3-1.7B Q4_0, weights from layer 0's attention
and gate projections, at N 8192, K 2048, T 256, this box:

| | |
|---|---|
| relative Frobenius error | 4.24e-3 |
| worst single element | 4.58e-3 relative |
| worst per-token cosine | 0.99999 |
| steady-state time per dispatch | 5.93 ms |
| implied rate | 1.44 TFLOP/s |

The rate matches the figure openflowlm-next records for this kernel, which is
the check that we are driving it as intended rather than accidentally doing
less work.

---

### HYBRID-PREFILL-BASELINE: Prefill speed figures are measured, not sampled

**Applies to:** npu-prefill-engine
**Verification:** manual

Any prefill throughput figure this project quotes shall come from the median
of repeated runs inside one warm process, at a stated prompt length, with the
warm-up runs discarded and reported separately.

This is a rule about measurement, not about the driver, and it exists because
the rule was broken once and cost the project a headline number. A single
prefill timed in a fresh process can be off by more than a factor of ten: some
launches compile the Vulkan shader pipelines before the first prefill can run,
and the next several prefills are still slow while the clocks ramp. Timing is
not deterministic, so there is nothing here to assert in a test; the procedure
below is the verification.

**Verification (manual):**

```
build.cmd
cd third_party\llama-b10944
gpu-prefill.exe ..\models\Qwen3-1.7B-Q4_0.gguf --lens 256,512,1024,2048 --iters 30 --decode 64
```

Accept the run and quote its medians when all of these hold:

- the standard deviation of the timed runs is under 5% of their mean at every
  length — above that, something else is using the GPU; close it and re-run
- decode is within a few tokens per second of 62 at short context, which is
  the figure on record for this box and model, and says the machine is in the
  state the other measurements were taken in
- median prefill time grows at least linearly with prompt length, which says
  the runs are doing the work and not hitting a cache

Quote the median, name the prompt length, and say it is prefill. A mean drags
in the spikes; an unlabelled figure gets read as decode.

**Measured 2026-09-18** on Qwen3-1.7B Q4_0, Radeon 890M, flash attention on,
ubatch 512, 8 warm-up and 30 timed runs per length. Three sweeps in three
separate process launches:

| prompt tokens | median ms | prefill tok/s | spread within a sweep (sd) | decode tok/s |
|---|---|---|---|---|
| 256 | 146.0 | 1754 | 4.1% | 62.6 |
| 512 | 295.2 | 1734 | 2.9% | 60.0 |
| 1024 | 604.0 | 1695 | 1.8% | 59.3 |
| 2048 | 1349.1 | 1518 | 3.8% | 53.9 |

The other two sweeps gave 1603 / 1699 / 1637 / 1520 and 1762 / 1811 / 1751 /
1623 tok/s at the same four lengths. So **medians reproduce across launches to
within about 10%, not better**, and the table above is the middle sweep. The
first of the three ran with other work on the machine and is the slow end.

Quote these as roughly 1700 tok/s at short prompts and roughly 1550 at 2048.
The extra digits are not real, and a comparison that turns on a 10% difference
needs the two things measured alternately in one session, not read off this
table.

Batch size barely matters. At 2048 tokens, varying the micro-batch over 128,
256, 512, 1024 and 2048 moved throughput only between 1560 and 1612 tok/s.
The GPU is limited by arithmetic here, not by how work is submitted, so a
comparison against an NPU working in 256-token blocks is not disadvantaged by
the block size.

**What this settles.** The withdrawn "about 1850 tokens per second" was
unquotable as measured — one sample, no repeats — but it was not far off. The
honest correction is smaller than the withdrawal implied: GPU prefill on this
box is roughly 1700 tok/s at short prompts and 1550 at 2048.

**What it compares against.** The NPU's projection-only ceiling from
HYBRID-NPU-GEMM is 511 tok/s, which would make the open kernels about three
times slower than the GPU at prefill. That comparison is arithmetic on one
GEMM kernel's rate and is weaker than what openflowlm-next3 has since measured
end to end (its PR #97, Qwen3-4B, a 981-token prompt, paired on one box): the
closed vendor engine reads that prompt at about 495 tok/s and the open kernels
at about 25. Adjusting for the larger model, the vendor's NPU prefill is in
the same range as this box's GPU, while the open kernels are more like thirty
times behind it.

So this baseline should not be read as "the NPU is three times slower." It is
the yardstick, and against it the hardware looks competitive while the open
kernels do not. The full argument is in the plan under "What openflowlm-next3
already measured".

---

### HYBRID-BFP16-NUMERICS: The NPU's 8-bit datapath leaves the GPU's answer as llama.cpp's own paths do
**Applies to:** npu-prefill-engine
**Verification:** manual

Prefill on the NPU shall use bfp16ebs8 matmuls: blocks of eight values sharing
one exponent, each with an 8-bit mantissa. The weights are converted when the
model loads, and the activations on the host, both rounded to nearest. The
keys and values that result, handed to a GPU context, shall disturb the GPU's
next-token predictions by no more than twice what the GPU disturbs them by when
it prefills the prompt itself.

This requirement exists because the NPU's fast matmul path is 8-bit only. It
has no native bf16 matrix instruction. The one conversion the NPU core offers
rounds down, and that measured six times worse than rounding to nearest, so
where the conversion happens is part of the contract.

**Verification (manual):**

```
build.cmd
cd third_party\llama-b10944
prefill-numerics.exe ..\models\Qwen3-1.7B-Q4_0.gguf --prompt-file ..\..\hybrid\prompts\prose.txt ^
    --prompt-file ..\..\hybrid\prompts\code.txt --prompt-file ..\..\hybrid\prompts\chat.txt ^
    --prompt-file ..\..\hybrid\prompts\data.txt --prompt-file ..\..\hybrid\prompts\long.txt --steps 32
```

It takes about 40 minutes on this machine. The tool prefills each prompt with
the host reference once in float32 and once per rounding mode, all from the
same 4-bit file. It hands each result to a GPU context and feeds the GPU the
float32 run's continuation, then compares every position against the float32
run.

**Acceptance criteria:**

- The "bfp16 nearest" row's mean KL is at most twice the "GPU prefills
  itself" row's.
- The "bfp16 nearest" row's top-1 agreement is within one percentage point of
  the GPU's own.
- When the NPU kernel exists, its keys and values replace the simulated ones
  and meet the same bar.

**Measured 2026-09-26** (Qwen3-1.7B Q4_0, five prompts, 165 positions):

| datapath | top-1 agrees | mean KL | max KL |
|---|---|---|---|
| GPU prefills itself | 99.4% | 8.3e-4 | 9.0e-3 |
| llama.cpp CPU rounding (8-bit activations, blocks of 32) | 100% | 1.1e-3 | 1.1e-2 |
| bf16 | 100% | 7.0e-4 | 1.9e-2 |
| bfp16, rounded to nearest | 98.8% | 1.6e-3 | 2.4e-2 |
| bfp16, activations rounded down (the core's conversion) | 97.0% | 1.0e-2 | 1.2e-1 |

**Result:** passes, with a mean KL ratio of 1.96. For scale, the 4-bit weights
themselves cost a mean KL of 0.14 against the full-precision model
(`llama-perplexity --kl-divergence`, same machine). The simulated rounding
matches mlir-aie's `floatToBfp16` / `bfp16ebs8ToFloat` bit for bit in the
round-down mode that helper implements.

---

## Below the traceability line

- The exact byte layout of the state (`hybrid/kv_state.h` documents it). It is
  the pinned release's and will move when the pin moves; the requirement is
  that our writer tracks it, not what it is.
- The convention that the handoff carries every prompt token but the last, and
  the GPU context runs that last token itself to obtain logits. A bare cache
  import has none. The real driver will do the same, and gets the first
  prediction as a by-product.
- Both contexts run with flash attention on so the value cache is stored one
  row per cell on both sides; the import checks that its layout matches the
  export's and refuses otherwise.
- First launch of a freshly built binary pays about 1.3 s of Vulkan pipeline
  compilation on this driver, cached for later runs. Discard the first run of
  any new build when timing.
- The host reference is float32 throughout and dequantises weights on the fly:
  about 3 s of fixed cost per run for reading and unpacking the weights, plus
  roughly 90 ms per prompt token on 24 threads. It is an oracle, not a path.
- The reference reads only the tied-embedding form of Qwen3 (no separate
  output head), which is what every Qwen3 GGUF up to 8B has.
- Which GGUF quantisations the packer accepts. `Q4_0` and `Q4_1` map to the
  chunk by rearrangement alone; `Q4_K`, which is what most published GGUFs
  use, does not, and is a separate piece of work. The requirement is that
  whatever is accepted is not requantised, not that everything is accepted.
- The band row split. It is the kernel's parameter, not ours; we produce
  whichever the kernel it feeds was built for.
- The activation tiling's parameters (64 by 32, with 8 by 8 inside). They come
  from the kernel, and the port in `hybrid/npu_gemm.cpp` follows it.
- Timing the NPU needs the first several dispatches thrown away. At ten
  iterations the warm-up still moved the best time from 5.9 ms to 9.5 ms, a
  60% error; at thirty it is stable to within 1%.
