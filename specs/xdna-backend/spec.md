# XDNA backend — requirements

Canonical spec for the ggml backend that offloads prefill matmuls to the AMD
XDNA2 NPU. Home repo: `ggml-xdna`.

The NPU kernel is `kernels/bfp16_gemm` in this repo: one core program
(`kernels/bfp16_gemm/prebuilt/bfp16_gemm.xclbin`) that serves every matrix
size, driven by an instruction stream the backend makes per size
(XDNA-INSTS-GEN). This spec covers how the backend presents itself to
`ggml_backend_sched`, what its matmul must produce, and that stream.

**Scope, revised 2026-09-29.** The backend now works next to the Vulkan GPU
instead of the CPU (plan: `.claude/plans/backend-size-aware.md`). It
registers as an integrated GPU, shares Vulkan's buffer type, and takes only
the prompt work its size rules accept; Vulkan runs everything else, and every
reply token. The earlier design (an accelerator device with its own host
buffer, next to the CPU) could not coexist with the GPU: llama.cpp ranks
accelerators below every GPU. XDNA-HOST-BUFT is retired for that reason.
Claiming whole blocks rather than single matmuls is a requirement of the new
design (the plan's step 0) and will add requirements here as it lands.

Tests: `specs/xdna-backend/tests/` is empty by design — the test binaries are
built by the top-level `CMakeLists.txt` from `tests/`, because they must link
the same import libs as the backend. Run them with `GGML_BACKEND_PATH` set:

```
build.cmd
set GGML_BACKEND_PATH=...\third_party\llama-b10944\ggml-xdna.dll
third_party\llama-b10944\test-dispatch-gate.exe
third_party\llama-b10944\test-mul-mat.exe
third_party\llama-b10944\test-safe-start.exe
build\test-insts-gen.exe
```

`test-dispatch-gate` and `test-mul-mat` run once as above (on the NPU) and
once more with `set GGML_XDNA_HOST_ONLY=1` (matmuls on the CPU reference; the
only mode on a machine without an NPU).

---

### XDNA-BACKEND-LOAD: Loads as an out-of-tree backend
**Applies to:** ggml-xdna
**Test category:** e2e

The backend shall build as a standalone DLL that a stock llama.cpp release
loads via the `GGML_BACKEND_PATH` environment variable, without any patch to
llama.cpp or ggml.

**Acceptance criteria:**
- `llama-bench --list-devices` with `GGML_BACKEND_PATH` set lists an `XDNA`
  device alongside the in-tree ones.
- Nothing in `third_party/llama-b10944` is modified except by dropping our own
  built files into it.

**Verification:** run `llama-bench.exe --list-devices` with the variable set and
confirm the `XDNA` row and the `load_backend: loaded XDNA backend` line.

---

### XDNA-HOST-BUFT: Weights stay readable by the CPU — RETIRED 2026-09-29

Retired with the move from working next to the CPU to working next to the
Vulkan GPU. An ACCEL device ranks below every GPU in llama.cpp's scheduler,
so it would never be handed an op. Replaced by XDNA-SHARED-BUFT.

---

### XDNA-SHARED-BUFT: Ops change hands with Vulkan without copies
**Applies to:** ggml-xdna
**Test category:** unit

The device shall register as `GGML_BACKEND_DEVICE_TYPE_IGPU`, report its
buffer type as the Vulkan device's own, accept Vulkan's buffer type in
`supports_buft`, and report no free memory (with a non-zero total).

Together these put every weight in Vulkan's memory and let the scheduler hand
an op to us or to Vulkan with no copies either way: we rank above Vulkan when
listed first in `-dev`, we can read its buffers, and llama.cpp gives a device
with no free memory no layers. Tensors in Vulkan buffers have no CPU pointer,
so the backend reads them with `ggml_backend_tensor_get_async` (the GPU
copies into host memory; the plain `ggml_backend_tensor_get` reads uncached
memory at 0.2 GB/s on this machine) and writes them with
`ggml_backend_tensor_set`.

It shall also accept host buffer types in `supports_buft`, and read tensors
in host memory directly. That is a correctness requirement, not a
convenience. Before a graph piece with no scheduler-copied inputs,
llama.cpp waits for the previous backend (Vulkan). Before a piece with
copied inputs, it waits only for the backends those inputs came from. A
piece whose one copied input came from the CPU (the token embeddings)
started reading Vulkan's results before Vulkan had finished them, and gave
plausible but wrong text (found 2026-09-29). Accepting host buffers means
our pieces never have copied inputs.

**Acceptance criteria:**
- `ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU`
- `ggml_backend_dev_buffer_type(dev) == ggml_backend_dev_buffer_type(Vulkan0)`
- `ggml_backend_dev_supports_buft(dev, Vulkan0's buffer type)` is true
- `ggml_backend_dev_supports_buft(dev, the CPU's buffer type)` is true
- `ggml_backend_dev_memory` reports free 0 and total above 0

---

### XDNA-BATCH-GATE: Only large batches are claimed
**Applies to:** ggml-xdna
**Test category:** unit

`supports_op` shall return false for `GGML_OP_MUL_MAT` whose batch dimension
(`ne[1]`) is below `GGML_XDNA_MIN_BATCH` (default 1024), and true at or above
it for the weights XDNA-SIZE-POLICY accepts.

This is the whole split between the NPU and the GPU for matmuls: reply
generation (batch 1) and short prompts fail the gate, and Vulkan runs them.
The default comes from the prototype, which beats the GPU from about 1,000
tokens; the plan's step 3 tunes it.

**Acceptance criteria:**
- batch 1 → not claimed
- batch 1023 → not claimed
- batch 1024, q4_K, 2048x2048 → claimed

---

### XDNA-CHUNK-WARNING: A warning when llama.cpp's chunks are too small
**Applies to:** ggml-xdna
**Test category:** manual

On the NPU, when llama.cpp's prompt chunk (`-ub`, 512 by default) is below
`GGML_XDNA_MIN_BATCH`, the backend shall log one warning naming the chunk
size and the flag to change (`-ub 2048`). Otherwise the backend silently
does nothing. The chunk size is read from the first matmul on loaded
weights with more than one row: llama.cpp lays out a prompt step at its
full chunk size before running anything. Matmuls on weights with no data
don't count; llama.cpp's memory fitting sets up a trial context that way,
with warnings hidden.

**Verification (manual):** Qwen3-1.7B, `-dev XDNA0,Vulkan0`:
- `llama-completion` and `llama-server` with the default `-ub` print the
  warning once;
- `-ub 2048` prints nothing;
- `-dev Vulkan0` prints nothing.

`llama-cli` hides all warnings unless `-v`, which the README says.

**Measured 2026-09-30:** as above.

---

### XDNA-WORK-FLOOR: Small matmuls stay on the GPU
**Applies to:** ggml-xdna
**Test category:** unit

`supports_op` shall also return false when the matmul is below
`GGML_XDNA_MIN_MFLOP` (default 256) MFLOP, computed as
`2 * src0->ne[0] * src0->ne[1] * batch`.

A dispatch costs something regardless of size, so a small matmul is cheaper
left where it is. Without this the old design claimed the per-layer
`ssm_alpha` / `ssm_beta` projections (36 KB weights), adding 80 dispatches per
prefill graph for a few MFLOP of work.

**Acceptance criteria:**
- q4_K 1024x64 at batch 1024 (134 MFLOP) → not claimed
- q4_K 2048x2048 at batch 1024 (8.6 GFLOP) → claimed

---

### XDNA-SIZE-POLICY: Weight matmuls, and the small ops next to them
**Applies to:** ggml-xdna
**Test category:** unit (matmuls), manual (block claiming)

Beyond the batch gate and the work floor, `supports_op` shall accept a
`MUL_MAT` only when its left side is a weight: a plain 2D tensor that is
neither a view nor computed, of a type ggml can unpack (any type with a
`to_float`). On the NPU (XDNA-SAFE-START), it shall also require the
weight's shape to fit the kernel: K a multiple of 64 and at
least 128, and N at most 32,768 once padded up to a multiple of 512 (the
backend pads the weight's NPU copy with zero rows and reads back only the
real columns).

On the NPU and with block claiming on (`GGML_XDNA_BLOCKS`, default on), it
shall also accept a small op when all of these hold: the backend's
executor implements it in that variant (RMS_NORM; MUL by a scale row; ADD of
two same-shape tensors; ROPE, normal or NeoX, with no YaRN ramp and no
frequency factors; SwiGLU, split and not swapped; RESHAPE); it covers at
least `GGML_XDNA_MIN_BATCH` tokens; and one of its inputs comes from an op
the policy accepts. It shall decline every other op.

Anything accepted is taken from Vulkan (the scheduler moves an op to the
higher-priority backend sharing its buffer type). Accepting too much
silently moves work, and accepting unrelated small ops would break Vulkan's
graph into pieces for nothing. The "input from an accepted op" rule grows
pieces out from the matmuls and stops there. Keys and values read from the
cache are views, so attention's own matmuls never match.

Once the NPU has failed during a run (XDNA-NPU-FAILURE), it shall decline
everything.

**Acceptance criteria (unit; in host-only mode, `GGML_XDNA_HOST_ONLY=1`, and
on the NPU):**
- batch 2048, q4_0 2048x2048 → claimed
- a matmul whose left side is a view of a larger tensor → not claimed
- a matmul whose left side is the result of another op → not claimed
- `ADD` of two leaf tensors at any size → not claimed

**Verification (manual, block claiming):** run with `GGML_XDNA_DUMP=1` and
check that each `xdna piece:` line on Qwen3-1.7B spans o-projection through
the next layer's k rope, and no piece contains an op outside the list above
other than views. Measured 2026-09-29: 29 pieces, each as described.

---

### XDNA-BLOCK-AGREES: Block claiming leaves the model's answer unchanged
**Applies to:** ggml-xdna
**Test category:** manual

With block claiming on, greedy generation after a prompt the backend reads
shall produce the same tokens as the GPU alone, to within the NPU's 8-bit
rounding. In practice that means the same continuation on a short prompt.

This is the end-to-end check on everything block claiming adds: the
executor's small ops, the fused SwiGLU, 16-bit outputs, the choice of which
results to write back, and the handoffs with Vulkan. A mistake in any of
them showed up as plausible but different text, never as a crash.

**Verification (manual):**
```
set GGML_BACKEND_PATH=...\ggml-xdna.dll
set GGML_XDNA_KERNELS=...\kernels\bfp16_gemm\prebuilt\bfp16_gemm.xclbin
set GGML_XDNA_MIN_BATCH=32
llama-completion -m Qwen3-1.7B-Q4_0.gguf -f <first 500 bytes of hybrid/prompts/prose.txt> -n 24 --temp 0 -no-cnv -fa on -b 2048 -ub 2048 -dev XDNA0,Vulkan0
```
and the same with `-dev Vulkan0`. The 24 generated tokens must match.

**Measured 2026-09-29:** identical ("…ides, the town was to be abandoned. The
people had to leave, and the quay was to be left as").

---

### XDNA-OPT-IN: Without being named, the backend changes nothing
**Applies to:** ggml-xdna
**Test category:** manual

With `GGML_BACKEND_PATH` set but without `-dev` naming `XDNA0`, llama.cpp
shall run exactly as if the backend weren't loaded. llama.cpp keeps only the
first integrated GPU it finds when no `-dev` is given, and Vulkan's
registers first.

**Verification (manual):** run `llama-completion` on the same prompt with
`--temp 0` three ways: without `GGML_BACKEND_PATH`; with it and no `-dev`;
with it and `-dev XDNA0,Vulkan0`. The first two must print "using device
Vulkan0" only and produce the same text; the third must produce the same text
too (XDNA-MUL-MAT-AGREES keeps it within rounding).

**Measured 2026-09-29** (Qwen3-1.7B Q4_0, a 105-token prompt,
`GGML_XDNA_MIN_BATCH=32`, `-b 2048 -ub 2048 -fa on`): the 24 generated tokens
are identical with `-dev Vulkan0` and with `-dev XDNA0,Vulkan0`. Reply
generation ran at 64–66 tok/s either way.

---

### XDNA-SAFE-START: No device unless the NPU can run the backend
**Applies to:** ggml-xdna
**Test category:** unit

When llama.cpp loads the backend, it shall offer the XDNA0 device only if
all of these hold:
- the kernel's xclbin is found: `GGML_XDNA_KERNELS` if set, else
  `bfp16_gemm.xclbin` next to `ggml-xdna.dll`;
- the NPU opens;
- it's a chip the backend was tested on (XRT's name "NPU Strix": Strix
  Point, Ryzen AI 300), unless `GGML_XDNA_ANY_NPU=1`;
- it loads the xclbin.

Otherwise it shall offer no device and log one line saying which check
failed, so llama.cpp runs as if the backend weren't there. The exception is
host-only mode (`GGML_XDNA_HOST_ONLY=1`), which offers the device and runs
claimed matmuls on a CPU reference, for tests and CI without an NPU. An
empty setting counts as unset.

Offering a device that can't run would have llama.cpp hand it work it then
fails, stopping the run.

**Acceptance criteria:** `tests/test-safe-start.cpp`: with a kernel path
that names nothing, the backend loads and XDNA0 is not offered.

**Verification (manual, the other checks):** on this machine, `llama-cli
--list-devices` with no settings lists XDNA0. The chip check was seen
refusing "NPU Strix" before it was added to the tested list, with the line
`xdna: not offering XDNA0: the NPU "NPU Strix" hasn't been tested with this
backend`. Measured 2026-09-30.

---

### XDNA-NPU-FAILURE: An NPU failure during a run doesn't stop it
**Applies to:** ggml-xdna
**Test category:** manual

If an NPU call fails during a run (starting it, building a weight's copy, a
submission, a wait), the backend shall:
- log one error line;
- finish the failed piece on the CPU, giving the same results within
  rounding (a failed piece has written nothing back, so it is run again);
- decline all work from then on, so later schedules go to the GPU.

Known limit: llama.cpp reuses a schedule for chunks of the same size, so
the rest of the current prompt keeps coming to the backend and runs on the
CPU, slowly. New prompts are scheduled afresh.

**Verification (manual):** `GGML_XDNA_FAIL_AFTER=n` makes every NPU
submission after the n-th fail.
- Qwen3-1.7B Q4_0, the 105-token prompt, `GGML_XDNA_MIN_BATCH=32
  -b 2048 -ub 2048`:
  - with n = 0 and n = 30, the run completes and logs one "the NPU failed"
    line;
  - with n = 0 the 24 tokens match the GPU's.
- `llama-perplexity --kl-divergence` against the GPU (6 chunks of 512):
  with n = 60, mean KL must be no worse than the NPU's without failures.

**Measured 2026-09-30:**
- The runs completed, n = 0 identical to the GPU.
- KL 0.0042 and 96.2% same top token, against 0.0081 and 95.3% without
  failures.
- The rest of that 6-chunk run took 65 s on the CPU, against 8 s on the
  GPU: the known limit.

---

### XDNA-NO-WEIGHT-TRANSFER: No weight is copied between backends
**Applies to:** ggml-xdna
**Test category:** manual

During prompt reading, the scheduler shall insert no copies into the
backend's graph pieces: every weight and activation it uses is read where it
lives, in Vulkan's memory.

**Verification (manual):** run with `GGML_SCHED_DEBUG=2 -v` and check that
every `## SPLIT #n: XDNA0` line reports `# 0 inputs`.

**Measured 2026-09-29** (same run as XDNA-OPT-IN): 984 ops assigned to XDNA0
across the scheduling passes, every XDNA0 split with 0 inputs; all layers'
weights assigned to Vulkan0.

---

### XDNA-NO-OFFLOAD-OP: No per-op weight copies
**Applies to:** ggml-xdna
**Test category:** unit

The device shall not implement `offload_op`.

That path exists for backends with their own memory: the scheduler copies the
weight into the backend's buffer for each offloaded op. Measured on this box,
the equivalent path for the iGPU made prefill 40% *slower* than plain CPU
(94 vs 157 tok/s) because of those copies. We only ever run on weights that are
already in our buffer.

**Acceptance criteria:**
- `ggml_backend_dev_offload_op(dev, op)` is false for a q4_K matmul at batch 512

---

### XDNA-MUL-MAT-AGREES: Matmul matches an fp64 reference
**Applies to:** ggml-xdna
**Test category:** unit

For every claimed weight type, the backend's `MUL_MAT` result shall be no
further from an fp64 reference — computed by dequantising the same weight bytes
and accumulating in double — than the CPU backend's result is, within a factor
of 2, and shall have cosine similarity above 0.9999 with that reference.

Bit equality against the CPU backend is the wrong bar: both quantise the
activation, differently. The CPU uses int8 blocks of 32 with a float scale;
the NPU uses 8-bit blocks of 8 with a shared power-of-two exponent and also
re-encodes the weight to 8 bits. The NPU's error comes out 1.3–1.7x the CPU's.
The factor was 1.5 until 2026-09-29, when the test first reached the NPU and
q4_0 measured 1.62x; the owner set 2 rather than pay for higher-precision
inputs (about half the matmul speed). The model-level check is
XDNA-BLOCK-AGREES.

**Acceptance criteria:**
- f32, f16, q8_0, q4_0, q4_K, q6_K, at shapes from 256x128x64 up to 2048x512x512
  (sizes with no NPU build: the host fallback)
- q4_0 2048x1024x512 and q4_K 2048x1024x600 with the weight in a weight
  buffer: on the NPU (on the CPU reference in host-only mode)
- the same at widths the kernel runs padded: q4_0 1536x896x512, q8_0
  1536x256x512, q4_K 1536x8960x520
- each case: `nrmse_xdna <= 2*nrmse_cpu + 1e-6` and `cos_xdna > 0.9999`

**Passing 2026-09-29**, on the NPU and in host-only mode, and with block
claiming on and off: the host cases at 1e-7 to 2e-7 against the CPU's 1e-7
to 7e-3; on the NPU, q4_0 8.7e-3 against 5.4e-3 (1.62x) and q4_K 8.9e-3
against 7.0e-3 (1.28x), cosine 0.99996. The padded cases, added the same
day, land at the same errors (8.7e-3 to 9.0e-3).

---

### XDNA-INSTS-GEN: The backend's NPU instruction streams match the toolchain's
**Applies to:** ggml-xdna
**Test category:** unit

For every size the kernel takes, the instruction stream the backend makes
(`hybrid/bfp16_insts.cpp`) shall equal, word for word, the `insts.bin` the
IRON toolchain builds from `kernels/bfp16_gemm/designs/whole_array_bfp_rtp.py`
(`-Tm 128 -Tk 64 -Tn 64 --c-tiled`, 8 columns) at that size and output mode.
Sizes the design can't take (M not a multiple of 512, N not a multiple of
512 or over 32,768, K not a multiple of 64 or under 128) shall be refused.

This is what lets the backend ship one ~200 KB xclbin and run any model's
sizes with no toolchain: the core program is the same for every size, and
only this stream changes. A wrong transfer descriptor can give plausible
wrong output rather than a crash, so it is checked word for word.

**Acceptance criteria:** `tests/test-insts-gen.cpp` against the reference
streams in `tests/insts/` (30 sizes: M 512 and 1,024; K 128 to 9,728; N 512
to 18,432; modes 0, 1 and 2), and five refused sizes. To add a size, build it
with `kernels/bfp16_gemm/build.ps1` and copy its `insts.bin` there.

**Passing 2026-09-29:** 30 of 30 identical, 5 of 5 refused.

---

---

### XDNA-DISPATCH-COST: Submission overhead stays under a millisecond
**Applies to:** ggml-xdna
**Test category:** manual

Per-dispatch submit+wait cost on this NPU shall be reported by
`tools/bench-dispatch.cpp`, which drives the vendored XRT shim
(`vendor/xrt-shim/`) against a built design. The backend's op-granularity offload depends on this staying small
against the per-split work.

Measured 2026-09-13: ~0.1 ms for designs from 75 to 3524 instruction words,
falling to 0.02-0.04 ms when runs are batched into a runlist. Note that the
~3.3 ms reported for FLM's fused whole-layer chains is a different quantity —
per-submission cost including instruction patching and BO syncs.

**Verification:** `build/bench-dispatch.exe <design-build-dir> <iters> <buffer
sizes...>`; the three rows are new-run-each, reused-run, and runlist.

---

## Below the traceability line

- The host reference matmul in `src/xdna-ref.cpp` is a placeholder for the NPU
  kernels. Its threading, blocking and dequant strategy are implementation
  detail; only XDNA-MUL-MAT-AGREES constrains it.
- The exact default values of `GGML_XDNA_MIN_BATCH` and `GGML_XDNA_MIN_MFLOP`
  are tuning, not contract. The requirements fix the mechanism, not the numbers.
- The backend's weight cache (step 1: raw bytes of each claimed weight, kept
  for tensors in buffers llama.cpp marks as weights, keyed on buffer, offset,
  type and shape) and the pinned host memory it reads activations into.
- `GGML_XDNA_TRACE=1`: a diagnostic that records the inputs of every op the
  backend declines and reports any of its own results no declined op was seen
  reading. It is the basis for deciding which results block claiming must
  write back.
- Which llama.cpp release we pin.
