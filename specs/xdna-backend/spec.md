# XDNA backend — requirements

Canonical spec for the ggml backend that offloads prefill matmuls to the AMD
XDNA2 NPU. Home repo: `ggml-xdna`.

The NPU kernel is `kernels/bfp16_gemm` in this repo: one core program
(`kernels/bfp16_gemm/prebuilt/bfp16_gemm.xclbin`) that serves every matrix
size, driven by an instruction stream the backend makes per size
(XDNA-INSTS-GEN). This spec covers how the backend presents itself to
`ggml_backend_sched`, what its matmul must produce, and that stream.

**Scope, revised 2026-09-29.** The backend works next to the Vulkan GPU
instead of the CPU. It registers as an integrated GPU, shares Vulkan's
buffer type, and takes only the prompt work its size rules accept; Vulkan
runs everything else, and every reply token. The earlier design (an
accelerator device with its own host buffer, next to the CPU) could not
coexist with the GPU: llama.cpp ranks accelerators below every GPU.
XDNA-HOST-BUFT is retired for that reason.

Tests: `specs/xdna-backend/tests/` is empty by design — the test binaries are
built by the top-level `CMakeLists.txt` from `tests/`, because they must link
the same import libs as the backend. `build.cmd` builds and runs them
(XDNA-BUILD). CTest labels them:
- `host`: no NPU needed. `test-dispatch-gate`, `test-memory-budget` and
  `test-mul-mat` run here with `GGML_XDNA_HOST_ONLY=1` (matmuls on the CPU
  reference), and need a Vulkan device.
- `npu`: `test-dispatch-gate`, `test-memory-budget` and `test-mul-mat` on
  the NPU.
- `nodriver`: `test-safe-start --no-driver`, on a machine without the NPU
  driver (CI).

`build.cmd` runs `host` and `npu` where the NPU driver is installed, and
`host` and `nodriver` elsewhere.

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
(`ne[1]`) is below `GGML_XDNA_MIN_BATCH` (default 1,024), and true at or
above it for the weights XDNA-SIZE-POLICY accepts. `npu --min-chunk N` sets
the variable (XDNA-LAUNCHER).

This is the whole split between the NPU and the GPU for matmuls: reply
generation (batch 1) and short prompts fail the gate, and Vulkan runs them.

Why 1,024: at llama.cpp's default prompt chunk (`-ub 512`) the NPU loses to
the GPU in every NPU power mode tried. Qwen3-1.7B, 2,048-token prompt, 20–22
paired runs each (2026-10-02): median 0.80x with the NPU's power mode on
"Default", 0.87x on "Performance", single runs 0.64–0.95x. At `-ub 2048` it
leads (1.06x and 1.15x). The kernel works in blocks of 512 rows, so a chunk
between 512 and 1,024 costs about what 1,024 does.

History: 1,024 until 2026-09-30, then 512 on a 5-round measurement that
showed a tie at 512 (0.98–1.07x). Those rounds happened to land on the
add-on's faster level every time (see XDNA-CHUNK-WARNING); back to 1,024 on
2026-10-02.

**Acceptance criteria:**
- batch 1 → not claimed
- batch 512 → not claimed
- batch 1,023 → not claimed
- batch 1,024, q4_K, 2048x2048 → claimed

---

### XDNA-CHUNK-WARNING: A warning when llama.cpp's chunks are too small — RETIRED 2026-09-30
**Applies to:** ggml-xdna

Added and retired the same day. The backend warned once when llama.cpp's
prompt chunk (`-ub`, 512 by default) was below `GGML_XDNA_MIN_BATCH`. The
owner dropped it in favour of README guidance (recommend `-ub 2048`). On an
idle machine the NPU only tied the GPU at 512-token chunks (0.98–1.07x), so
the default costs nothing; it just forgoes the 13–18% that 2,048-token chunks
give. Lesson kept for any future message of this kind: llama.cpp's memory
fitting sets up a trial context with warnings hidden and no weight data, and
a once-only message fired there is never seen.

Correction, 2026-10-02: the tie at 512 didn't hold up. The add-on's speed
lands at one of a few levels per run, about 20–25% apart, and the 5 rounds
behind "0.98–1.07x" all landed on the faster one. Over 20+ paired runs it
loses at 512 (0.80x); see XDNA-BATCH-GATE.

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
least `GGML_XDNA_MIN_BATCH` tokens; one of its inputs comes from an op
the policy accepts; and none of its inputs has its shape and may sit in
host memory (a graph input, a `GET_ROWS` result such as the token
embeddings, or a tensor already in a host buffer). It shall decline every
other op.

The last condition is there because ggml's allocator may put an op's result
in the memory of a same-shape input (in place) without checking that the
input is in the same kind of memory. An input in host memory would take the
backend's result there, and Vulkan, reading it next as one of its own
buffers, crashes. LFM2's first residual add, `add(embeddings, x)`, did
(2026-10-02).

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
- `ADD` of two claimable matmuls (batch 2048) → claimed on the NPU; not
  claimed host-only (no block claiming there)
- `ADD` of a claimable matmul and the token embeddings (`GET_ROWS`), in
  either order → not claimed
- `ADD` of a claimable matmul and a graph input → not claimed

**Verification (manual, block claiming):** run with `GGML_XDNA_DUMP=1` and
check that each `xdna piece:` line on Qwen3-1.7B spans o-projection through
the next layer's k rope, and no piece contains an op outside the list above
other than views. Measured 2026-09-29: 29 pieces, each as described.
Again 2026-10-03, after the host-memory condition: 30 pieces per chunk.
Layer 0's first norm and residual add read the token embeddings and stay on
the GPU, so layer 0 comes in three pieces (its q/k/v projections, its
o-projection, then its feed-forward through layer 1's k rope); every other
piece as described.

---

### XDNA-MEMORY-BUDGET: The NPU's weight copies are held to a memory limit
**Applies to:** ggml-xdna
**Test category:** unit (the decisions), manual (the log line)

The NPU keeps its own copy of every weight it uses, on top of llama.cpp's:
9 bytes for every 8 values, with the weight's width padded to a multiple of
512 (about 1.1 GB per billion parameters). Beyond XDNA-SIZE-POLICY,
`supports_op` shall accept a weight's matmul only while the copies of the
weights accepted so far, plus this one, fit a limit. It shall take weights in
the order llama.cpp asks about them (layer by layer, when it plans the first
prompt at load). When the first weight doesn't fit, it shall stop taking
new weights, and hand back to the GPU the weights already taken from that
weight's layer (named `blk.N.` by llama.cpp), so no layer is split, unless a
prompt has already run. Its answer for a weight shall then never change
while a context is alive. When the last context is freed (its copies go
with it), the count shall start again from nothing.

Weights with no data (llama.cpp's memory-fitting trial and its load-time
checks), weights outside a buffer llama.cpp marks as weights, and questions
asked with no context alive shall not count. Host-only mode
(`GGML_XDNA_HOST_ONLY=1`) shall decide exactly as the NPU would.

The limit is the memory available when the first weight is asked about,
less what's kept back for llama.cpp's work buffers and the rest of the
machine: the larger of 4 GB and a tenth of the machine's memory, and at most
20 GiB. `GGML_XDNA_MAX_COPY_GB` (in GiB, fractions allowed) sets it instead,
with no cap.

Without a limit, a big model's copies pass the machine's memory and Windows
starts paging, which looks like the add-on being slow, not like a memory
problem.

Why at most 20 GiB (added 2026-10-04): the NPU holds only so much. On the
88 GB test machine its memory in use flattens at about 25.9 GiB, and copies
past that come back damaged, silently: Qwen3.8-27B with all 64 layers on
the NPU (27 GB of copies) answered `////`, and with copies capped at 12, 16,
20 or 24 GB it answered right (2026-10-03; `sweep-crash-fixes.md`). A
standalone repro (`tools/npu-ceiling-repro.cpp`, branch
`tools/npu-ceiling-repro`, 2026-10-04, 3 runs) puts the ceiling at 26.0 GiB
of NPU buffers in all: fine at 26.0, broken at 27.9. It's the NPU's own:
a llama-server holding 14.8 GiB on the GPU didn't lower it. Broken buffers
read as 100% NaN on the NPU, with no error from the driver, while the CPU
still sees their bytes intact. The 20 GiB cap leaves room for the add-on's
other NPU buffers and for other programs using the NPU. Why the ceiling is
where it is isn't known; XDNA-NPU-RESULT-CHECK catches it when it's passed.

One budget covers the process, since `supports_op` is asked without a
context. Two contexts alive at once on the same model would each build the
copies the budget counted once; llama.cpp's tools and server don't do that
(the router runs one process per model).

**Acceptance criteria** (`tests/test-memory-budget.cpp`, with
`GGML_XDNA_MAX_COPY_GB=0.05`, 51.2 MiB; weights are 2048x2048 q4_0, a
4.5 MiB copy each, four to a layer, in a Vulkan buffer marked as weights;
in host-only mode and on the NPU):
- with no context started, a weight is claimed and nothing is counted
- with a context: a 2048x32768 weight with no data (72 MiB if counted) is
  claimed
- layers 0 and 1 (36 MiB) are claimed; layer 2's first three weights are
  claimed when asked (49.5 MiB), its fourth (54 MiB) is not
- layer 2's first three are then not claimed either
- a 2048x512 weight in layer 3 (1.1 MiB, which would fit) is not claimed
- layer 0's and layer 1's weights are still claimed
- after the context is freed, a new context claims the layer 3 weight and
  layer 2's fourth

**Verification (manual, the log line):** when the limit is reached, the
add-on shall log one warning saying the limit, where it came from, and how
many layers went to the NPU; when everything fits, one info line (shown
with `-v`) with the total. Run `npu llama-completion` on Qwen3-1.7B Q4_0
with a prompt over 512 tokens and `-n 24 --temp 0`, with
`GGML_XDNA_MAX_COPY_GB` set to 1, 0.01 and 100, and once unset with `-v`.
Compare each reply with `llama-completion -dev Vulkan0` on the same
arguments.

**Passing 2026-10-01:** 1 GB: `limited to 1.0 GB (set by
GGML_XDNA_MAX_COPY_GB): the first 18 layers on the NPU, the rest on the
GPU`. 0.01: `no layer fits, so all run on the GPU`. 100: `1.5 GB for 196
weights`. Unset: `limit 12.7 GB: 21.5 GB free, less 8.8 GB kept back`.
All four replies identical to the GPU's. `llama-bench -ub 512,1024 -v` at
1 GB: each test's new context printed the same 18-layer split.

---

### XDNA-COPY-AT-LOAD: The NPU's weight copies are built while the model loads
**Applies to:** ggml-xdna
**Test category:** manual

When llama.cpp plans a context's pieces (it does so at load, before the
first prompt, and shows each backend its pieces through `graph_optimize`),
the add-on shall start building the NPU copies those pieces will use, on a
thread of its own. A piece shall wait for that thread before it runs. The
copies built shall be the ones the pieces use, so none is built inside a
piece afterwards. Pieces whose weights have no data (llama.cpp's
memory-fitting trial) shall build nothing. A copy that fails to build shall
fail the NPU as XDNA-NPU-FAILURE describes. Freeing the context shall stop
the building after the copy in progress. `GGML_XDNA_COPY_AT_LOAD=0` shall
leave the copies to the first prompt, as before.

Building the copies takes seconds (1.9 s for Qwen3-1.7B and 4.4 s for
Qwen3-4B, measured on an idle machine in A2), and the first prompt paid for
all of it. A server loads and then waits for requests, so its copies are
ready by the first one. A tool that sends its prompt straight after loading
(`llama-completion`, `llama-cli -p`) still waits for most of them.

**Verification (manual):** with `GGML_XDNA_TRACE=1` on Qwen3-1.7B Q4_0,
where the trace line `xdna weight copies:` counts copies built at load and
inside pieces, and the time a piece waited:
1. `npu llama-server`; 30 s after it's listening, one `/completion` with a
   ~3,000-token prompt, `n_predict` 24, temperature 0. Copies: all built at
   load, 0 inside pieces, about 0 ms waited.
2. The same with `GGML_XDNA_COPY_AT_LOAD=0`: 0 at load, all inside pieces;
   the reply identical to step 1's.
3. `npu llama-completion` on the same prompt, `-n 24 --temp 0`: 0 inside
   pieces; the reply identical to `llama-completion -dev Vulkan0`.
4. `npu llama-bench -ub 512,1024 -r 1`, and `npu llama-completion` on a
   prompt under 512 tokens (copies still building at exit): both finish and
   exit normally.

**Passing 2026-10-01** (machine busy, CPU 83%): 1. 166 built at load, 0
inside pieces, 0.0 ms waited. 2. 0 at load, 166 inside; reply identical to
1's. Both differ from a GPU-only server by one token ("sincpf" for
"sincpi"), with building at load on or off, so the difference is the NPU's,
not this. 3. identical to the GPU; 0 inside pieces; waited 7.3 s. 4.
both exited normally.

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
set GGML_XDNA_MIN_BATCH=32
llama-completion -m Qwen3-1.7B-Q4_0.gguf -f <first 500 bytes of tests/prompts/prose.txt> -n 24 --temp 0 -no-cnv -fa on -b 2048 -ub 2048 -dev XDNA0,Vulkan0
```
and the same with `-dev Vulkan0`. The 24 generated tokens must match.

**Measured 2026-09-29:** identical ("…ides, the town was to be abandoned. The
people had to leave, and the quay was to be left as"). Again 2026-09-30,
after the repository cleanup: identical, 269 pieces on the NPU. Again
2026-10-03, with small ops next to host memory declined (XDNA-SIZE-POLICY):
identical.

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

When llama.cpp loads the backend, it shall load whether or not the NPU
driver is installed, and offer the XDNA0 device only if all of these hold:
- the kernel's xclbin is found: `GGML_XDNA_KERNELS` if set, else
  `bfp16_gemm.xclbin` next to `ggml-xdna.dll`;
- the NPU driver's `xrt_coreutil.dll` is found and exports, by name, every
  function the backend uses (`vendor/xrt-implib/xrt_coreutil.def`). The DLL
  is delay-loaded, and all of its functions are bound here, so a missing one
  turns the backend off instead of failing at a call;
- the NPU opens;
- it loads the xclbin;
- it gets a test multiply right: 512 x 128 times 128 x 512, of small whole
  numbers the kernel holds and adds exactly, every answer equal to the same
  multiply done on the CPU, within 5 seconds.

There is no list of chips: any NPU that passes the test is taken. Only Strix
Point (XRT's name "NPU Strix") has been tested. Strix Halo and Krackan Point
have the same NPU and should pass.

Otherwise it shall offer no device and log one line saying which check
failed, so llama.cpp runs as if the backend weren't there. The exception is
host-only mode (`GGML_XDNA_HOST_ONLY=1`), which offers the device and runs
claimed matmuls on a CPU reference, for tests and CI without an NPU. An
empty setting counts as unset.

Offering a device that can't run would have llama.cpp hand it work it then
fails, stopping the run.

Known gap (found 2026-09-30): with no device offered, a command that names
`-dev XDNA0,Vulkan0` itself stops at llama.cpp's argument check
(`invalid device: XDNA0`) rather than running on the GPU. The launcher
(XDNA-LAUNCHER) avoids it by naming XDNA0 only when it's offered.

**Acceptance criteria:** `tests/test-safe-start.cpp`:
- with a kernel path that names nothing, the backend loads and XDNA0 is
  not offered;
- `--no-driver`, on a machine without the NPU driver (CI): with the real
  kernel, the backend loads and XDNA0 is not offered.

**Verification (manual, the other checks):** on this machine, `llama-bench
--list-devices` with no settings lists XDNA0 and logs `xdna: NPU "NPU
Strix", ..., test multiply right (20 ms)` (19-22 ms over 13 starts). With
`GGML_XDNA_KERNELS` naming an older build that loads but never finishes
(`kernels/bfp16_gemm/build/whole_array_bfp_acc_bf16d2/
1024x2048x2048_128x64x64_c8`), it logs `not offering XDNA0: the NPU "NPU
Strix" couldn't run a test multiply (no answer within 5000 ms)`, llama-bench
exits normally 9.3 s after starting, and the next start passes again.
Measured 2026-10-02. The wrong-answer path has not been seen on hardware.

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

### XDNA-NPU-RESULT-CHECK: Damaged NPU results are caught, not passed on
**Applies to:** ggml-xdna
**Test category:** unit

Every value the backend reads back from the NPU shall be checked for NaN and
infinity as it's decoded. If one is found, that piece (or matmul) shall
count as an NPU failure under XDNA-NPU-FAILURE: one error line saying the
NPU's results held NaN or infinity, the piece run again on the CPU before
anything is written back, and the GPU for everything after.

Why: past the NPU's memory ceiling (XDNA-MEMORY-BUDGET), Windows moves the
NPU's buffers out and they come back damaged, with no error. A damaged
weight copy turns every result that uses it into NaN, and everything after
it in the model. Without the check the reply is garbage (`////`) and
nothing says why. The check is a bit test on values the decoder already
reads. Damage that gives finite wrong values isn't caught; none has been
seen.

**Acceptance criteria** (`tests/test-mul-mat.cpp`, run as `mul-mat-npu-nan`
on the NPU with `GGML_XDNA_NAN_AFTER=0`, which turns every NPU result to NaN
once it's back):
- every case passes against the fp64 reference, as in `mul-mat-npu`

Before the check (2026-10-04) the five cases the NPU takes failed there with
NaN; with it, all pass.

**Verification (manual, block claiming):**
- Qwen3-1.7B Q4_0, a prompt over 1,024 tokens, `-n 24 --temp 0`, with
  `GGML_XDNA_NAN_AFTER=20`: the run completes, logs one `the NPU failed (the
  NPU's results held NaN or infinity ...)` line, and the reply matches the
  GPU's.
- Qwen3.8-27B UD-IQ3_S, the 3,846-token prompt: with `npu --memory-gb 30`
  (all 64 layers, past the ceiling) the line appears and the reply is
  sensible, not `////`; with the default limit (20 GiB) no line, and a
  sensible reply.

**Measured 2026-10-05**, against a build of main without the check
(2efeefe), one run each unless noted, the NPU otherwise unused:
- Qwen3-1.7B, `NAN_AFTER=20`: one line; the reply is the GPU's, word for
  word.
- Qwen3.8-27B:

  | build, limit | NPU copies | reply | prompt time |
  |---|---|---|---|
  | without the check, `--memory-gb 30` | all 64 layers | `////`, no error | 59 s |
  | with it, `--memory-gb 30` | all 64 layers | the line at 47 s, then the GPU's reply | 1,304 s |
  | with it, default | 49 layers (20.0 GB) | the GPU's reply, no line | 70 s |

  The 1,304 s is XDNA-NPU-FAILURE's known limit: the rest of that prompt
  ran on the CPU.
- Cost of the check: Qwen3-1.7B, `llama-bench -p 2048 -ub 2048 -b 2048
  -r 3`, 6 rounds alternating the two builds. Time in the host passes,
  where the check runs: 295.5 ms per trace line with it, 295.1 without.
  Prompt speed: median 0.98x, single rounds 0.97-1.06x, within the
  round-to-round spread.

---

### XDNA-MODEL-WARNINGS: A model the NPU suits badly is warned about, not refused
**Applies to:** ggml-xdna
**Test category:** unit

If the user asked for the NPU, the NPU runs the model. Where the backend knows
the NPU is a poor fit it shall say so once and carry on; it shall not hand the
model back to the GPU on its own. Two cases:

- **Mixture of experts.** A `MUL_MAT_ID` anywhere in the graph marks one. The
  backend shall not claim that op, shall still claim the model's ordinary
  matmuls, and shall warn that the NPU cannot take the expert step, that these
  models' answers drifted further from the GPU's than any dense model's in our
  tests, and that `-dev Vulkan0` would use the GPU alone.
- **A narrow model.** Width under 2,048, read from the first dimension of
  `token_embd.weight` or of a layer's `attn_q.weight` / `attn_qkv.weight`. The
  backend shall warn that the GPU alone was faster on every model this size we
  tested.

Each warning is said at most once per model, from `graph_compute` rather than
from `supports_op`, because llama.cpp's memory fitting plans graphs with
warnings hidden but never computes one — the lesson of the retired
XDNA-CHUNK-WARNING. The record resets when the last context goes, so a second
model in the same process gets its own warnings.

Why: the owner's rule, 2026-10-04 — "if the user asks to run it on the NPU by
using our utility, we should respect that; for the known bad models at best we
should output a warning". It replaces the earlier XDNA-MOE-DECLINE idea, which
would have refused these models outright. The evidence behind each warning is
the model sweep: LFM2.5 8B-A1B and gpt-oss-20b reached KL 0.020 and 0.037
against a 0.01 bar, while every dense model stayed under it, most likely
because rounding each layer's router multiply to 8 bits flips which experts a
token uses; and models narrower than 2,048 ran at 0.34-0.84x the GPU, while
2,048 ties.

**Acceptance criteria** (`tests/test-dispatch-gate.cpp`, both modes):
- a `MUL_MAT_ID` (experts `[2048, 2048, 8]`, rows `[2048, 1, 2048]`, ids
  `[1, 2048]`) is not claimed
- an ordinary q4_0 2048x2048 matmul at batch 2,048 is still claimed after that

These guard the rule rather than the warning: they pass on the build before
the warnings too, and fail if the backend is ever made to decline a
mixture-of-experts model.

**Verification (manual):** a prompt over 1,024 tokens through `npu
llama-completion` (which passes `-ub 2048`), `-n 8 --temp 0`:
- a mixture-of-experts model: the mixture-of-experts line appears exactly
  once, the run finishes, and the reply is sensible
- a model of width under 2,048: the narrow line appears exactly once, with the
  right width
- Qwen3-1.7B (width 2,048): neither line, and `GGML_XDNA_TRACE=1` shows NPU
  pieces, so the silence is the width rule and not an idle backend
- the same narrow model through `npu llama-server`: no line while it loads,
  then the line on the first real prompt

**Measured 2026-10-05**, a 3,175-3,246-token prompt from llama.cpp's docs:

| model | width | lines |
|---|---|---|
| qwen2.5-1.5b-instruct Q4_0 | 1,536 | narrow x1 |
| Qwen3-1.7B Q4_0 | 2,048 | none; 196 matmuls in 31 NPU pieces |
| gpt-oss-20b MXFP4 (mixture of experts) | 2,880 | mixture-of-experts x1; ran to the end |
| qwen2.5-1.5b through `llama-server` | 1,536 | none at load, narrow x1 on the first prompt |

---

### XDNA-SERVER-USE: Works with llama.cpp as people run it
**Applies to:** ggml-xdna
**Test category:** manual

With llama.cpp's default settings plus `-dev XDNA0,Vulkan0`, the backend
shall neither crash nor change the model's answers beyond XDNA-BLOCK-AGREES'
bar in these uses:
- several prompts' rows in one chunk (llama-server's parallel slots and
  continuous batching);
- a prompt that continues a cached conversation;
- contexts up to 32,768 tokens;
- llama-server's router with several models loaded at once. The router
  runs each model in its own process, so this is several processes using
  the NPU at the same time.

Every op the backend runs treats each row (token) on its own, so which
prompts share a chunk can't change a result. Each process has its own
weight copies, freed with its llama.cpp context.

**Verification (manual):** Qwen3-1.7B Q4_0. The text is any long file
that doesn't repeat itself; 2026-09-30 used llama.cpp's own docs
concatenated (`docs`, `tools` and `examples` `*.md`, 265k tokens).
- `llama-perplexity -f <text> -c 512 --chunks 16 -b 2048`, first with
  `-dev Vulkan0 --kl-divergence-base base.kld`, then with
  `-dev XDNA0,Vulkan0 --kl-divergence-base base.kld --kl-divergence` at
  `-ub` 512 and 2048. At `-ub 2048` each chunk holds 4 prompts. The two
  must give the same KL, in the range every model so far has shown: mean
  KL under 0.01 and at least 95% same top token.
- The same at `-c 32768 --chunks 1` (the base file is ~5 GB).
- `llama-server -np 4 -c 16384`, once with `-dev Vulkan0` and once with
  `-dev XDNA0,Vulkan0`. Send 12 prompts of 700–3,000 tokens at once, and 6
  conversations of a prefix and then the prefix plus ~1,200 tokens
  (`cache_prompt`, one slot each), with `n_predict` 16, `temperature` 0,
  `n_probs` 10. The first generated token must match in every request.
- The router: `llama-server --models-preset <ini with two models> -np 4
  -c 16384`, 12 prompts to each model at the same time. The same check,
  and `GGML_XDNA_TRACE=1` shows both child processes' NPU pieces.

**Measured 2026-09-30:**
- 4 prompts per chunk: KL 0.0052, 96.2% same top token, identical at
  `-ub` 512, 1,024 and 2,048.
- 32k context: KL 0.0036, 97.9%, identical at `-ub` 512 and 2,048.
- Server: first token the same in 23 of 23 requests; 19 of 23 16-token
  replies identical word for word. The rest split at near-ties
  ("--model-name" against "--model_name").
- Router, Qwen3-1.7B and Qwen3-4B Q4_K_M: first token the same in 44 of
  44; 38 of 44 replies identical; both processes on the NPU throughout.
- No crash in any run.

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
(`src/npu/bfp16_insts.cpp`) shall equal, word for word, the `insts.bin` the
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
with `kernels/bfp16_gemm/build.ps1 -M <M> -K <K> -N <N> [-OutMode <mode>]`
and copy its `insts.bin` there.

**Passing 2026-09-29:** 30 of 30 identical, 5 of 5 refused.

---

### XDNA-BFP16-PACK: Weights are laid out as the kernel was verified with
**Applies to:** ggml-xdna
**Test category:** unit

The packer that builds each weight's NPU copy (`src/npu/bfp16_pack.cpp`)
shall produce, byte for byte, what the host code the kernel was verified
with produces: blocks of eight values sharing one exponent byte, rounded to
nearest even, in the kernel's tile order, for both operands and several
tilings. Unpacking shall give back exactly the decoded values.

A wrong layout gives plausible wrong output, not a crash, and on a machine
without an NPU this is the only check of it (XDNA-MUL-MAT-AGREES covers it
end to end on the NPU).

**Acceptance criteria:** `tests/test-bfp16-pack.cpp` against that
reference, copied into the test unchanged (its source,
`kernels/bfp16_gemm/bench_bfp16.cpp`, is in the `research-archive` tag).

---

### XDNA-BUILD: One command builds and tests it
**Applies to:** ggml-xdna
**Test category:** manual

On Windows with Visual Studio 2022 (or its Build Tools) and the C++ tools,
`build.cmd` in a clean checkout shall download the pinned llama.cpp release,
build the backend and its tests, and run the tests the machine can run, with
no NPU driver needed to build.

**Verification (manual):** the CI workflow (`.github/workflows/ci.yml`) runs
`build.cmd` on every pull request, on a GitHub Windows machine with no NPU
or GPU (a software Vulkan device stands in), so it runs the `host` and
`nodriver` tests. On this machine, `build.cmd` runs `host` and `npu`.

**Passing 2026-09-30:** on this machine, 7 of 7 tests; in CI, 6 of 6
(`host` and `nodriver`). CI sets `GGML_XDNA_PINNED=0`: on Windows,
lavapipe's pinned memory fails ggml's alignment check.

---

### XDNA-LAUNCHER: `npu` runs llama.cpp with the NPU, with nothing to set
**Applies to:** ggml-xdna
**Test category:** manual

`npu <llama.cpp program> [arguments]` (`tools/npu.cpp`, built next to the
llama.cpp programs) shall run that program so that:
- llama.cpp loads the add-on next to `npu.exe`, unless `GGML_BACKEND_PATH`
  is already set;
- when llama.cpp offers XDNA0, `-dev XDNA0,Vulkan0` is added
  (`XDNA0/Vulkan0` for llama-bench), then `-ts 0,1` (not for llama-bench)
  unless the user set a split (`-ts`, `--tensor-split` or
  `LLAMA_ARG_TENSOR_SPLIT`), and `-ub 2048 -b 2048` for llama-server,
  llama-cli and llama-completion. When it doesn't, nothing is added,
  llama.cpp runs on the GPU, and one line gives the add-on's reason;
- with no model given (no `-m`, `-hf` and the like, nor their
  `LLAMA_ARG_*` variables) and a console to ask on, it lists the `.gguf`
  models in a `models` folder next to it, LM Studio's folder, llama.cpp's
  download folder and the Hugging Face download folder (vision add-ons and
  parts 2+ of split models left out), and runs the one picked. Enter picks
  the last one. For llama-server, "all of them" runs the router over every
  model listed;
- whatever the user set (`-dev`, `-ub`, `-b`, the model, or their
  `LLAMA_ARG_*` variables) is kept, and the rest of the command line
  reaches the program exactly as typed;
- Ctrl+C reaches the program, and `npu` returns the program's exit code. If
  `npu` is closed or killed, the program goes with it.

`npu`'s own options come before the program's name, as
`--name value` or `--name=value`:
- `--memory-gb N` (a number, 0 or more; fractions allowed) shall set
  `GGML_XDNA_MAX_COPY_GB` to N for the program, replacing any value already
  set (XDNA-MEMORY-BUDGET), and the `npu: prompts on the NPU (...)` line
  shall end with `; NPU weight copies limited to N GB`;
- `--min-chunk N` (a whole number, 1 or more) shall set
  `GGML_XDNA_MIN_BATCH` to N for the program, replacing any value already
  set (XDNA-BATCH-GATE), and the `npu:` line shall end with
  `; NPU takes chunks of N tokens or more`;
- a bad or missing value, or an option `npu` doesn't know, shall stop `npu`
  with one line saying what's wrong, before any program starts, and exit 1;
- `-h` or `--help` there shall print the usage, options included.

They go before the name so they never mix with the program's own options:
llama.cpp's programs reject options they don't know, and llama.cpp passes
add-ons no settings of their own. llama-server's router copies its
environment into every model's server, so the option covers them all.

The launcher is what makes "run it with nothing to set" true. It also
avoids XDNA-SAFE-START's gap: a command naming XDNA0 when there isn't one.

`-ts 0,1` gives XDNA0 no layers, which is what llama.cpp picks anyway (it
splits layers by free memory, and XDNA0 reports none). Said outright, it
stops llama.cpp's memory fitting (`common/fit.cpp`) after the context size
and before its layer search. That search divides by the change in a
device's memory use as layers move onto it, which for XDNA0, sharing the
GPU's memory, is always 0: mixture-of-experts models reach it and crashed
while loading (2026-10-02). llama-bench doesn't fit unless asked.

**Verification (manual):**
- From cmd, `npu llama-echo -p "a & b | c * d! e" -x "say \"hi\"" --path
  "C:\dir with space\\" plain*.gguf ^& "" last`, with a program that
  prints its arguments standing in for llama-echo: every argument arrives
  unchanged, and the exit code comes back.
- `npu llama-completion -m Qwen3-1.7B-Q4_0.gguf -f <XDNA-BLOCK-AGREES'
  prompt> -n 24 --temp 0 -no-cnv -fa on` with `GGML_XDNA_MIN_BATCH=32`:
  prints `npu: prompts on the NPU (added -dev XDNA0,Vulkan0 -ts 0,1
  -ub 2048 -b 2048)`, and the 24 tokens match the GPU's.
- The same with `GGML_XDNA_KERNELS` naming nothing: prints
  `npu: running on the GPU only: ...` and runs.
- With `-dev Vulkan0`: nothing added. With `-ub 512`: only `-dev` and
  `-ts` added. With `-ts 1,1`: no `-ts` added. `npu llama-bench`:
  `-dev XDNA0/Vulkan0` only.
- `npu llama-completion` on gpt-oss-20b and LFM2.5-8B-A1B (mixture of
  experts): loads, and the `npu:` line shows `-ts 0,1`.

  Measured 2026-10-03: as listed, for `-ts 1,1`, `--tensor-split 1,1`,
  `LLAMA_ARG_TENSOR_SPLIT=1,1`, `-ub 512` and llama-bench; gpt-oss-20b,
  LFM2.5-8B-A1B and Ornith 35B (Qwen3.6 35B-A3B) load and finish (all three
  crashed while loading before).
- In a console (a script can drive one with a pseudo console): with no
  model, the list shows; a number runs that model; next time Enter runs it
  again; 0 for llama-server lists every model in the router's `/models`,
  and a request to one loads it.
- llama-server through `npu`, Ctrl+C in its console: the server cleans up
  and exits 0, and no llama-server process is left.

- The first check again with `--memory-gb 20` and with `--memory-gb=0.5`
  in front of `llama-echo`, the stand-in also printing
  `GGML_XDNA_MAX_COPY_GB`: the same arguments arrive, the variable is 20
  and 0.5, the exit code comes back. `npu --memory-gb 1 llama-completion`
  on the second check's prompt: the `npu:` line ends with `; NPU weight
  copies limited to 1 GB`, the add-on's warning follows, and the tokens
  match the GPU's. `--memory-gb lots`, `--memory-gb -1`, `--memory-gb`
  alone and a misspelled option: one line each, exit 1, nothing started.
- The first check again with `--min-chunk 512` and with
  `--min-chunk=2048 --memory-gb 20`, the stand-in also printing
  `GGML_XDNA_MIN_BATCH`: the same arguments arrive, the variables are as
  given, the exit code comes back. With `GGML_XDNA_MIN_BATCH=4096` already
  set, `--min-chunk 512` gives 512. `npu --min-chunk 32 llama-completion`
  on the second check's prompt (105 tokens): the `npu:` line ends with
  `; NPU takes chunks of 32 tokens or more`, the add-on builds its weight
  copies (`-v` shows the line), and the tokens match the GPU's; without the
  option, no copies are built. `--min-chunk lots`, `0`, `1.5`, `-1`,
  `--min-chunk` alone and `--min-chunks 512`: one line each, exit 1, nothing
  started.

**Passing 2026-09-30:** all of the above but `--memory-gb` (added
2026-10-01). The menu listed 38 models from LM Studio and the Hugging Face
cache; the router listed 41 (llama.cpp adds 3 of its own).

**Passing 2026-10-01 (`--memory-gb`):** all of its checks. Arguments
identical with the option in front, without it, and run directly; exit
code 7 came back each time. `--memory-gb 1`: 18 of 28 layers on the NPU,
reply identical to the GPU's; `--memory-gb=0.01`: no layer, identical.
The router's copying of its environment was checked in its source
(`tools/server/server-models.cpp`, b10944), not by a run.

**Passing 2026-10-02 (`--min-chunk`):** all of its checks. Qwen3-1.7B
Q4_0: with `--min-chunk 32`, 1.5 GB of copies for 196 weights and a reply
identical to the GPU's; without it, no copies and the same reply.

---

### XDNA-RELEASE: A tag builds the release zip in CI
**Applies to:** ggml-xdna
**Test category:** manual

Pushing a tag `v*` shall run `.github/workflows/release.yml` on a GitHub
Windows machine, which builds and tests as XDNA-BUILD does, then makes
`ggml-xdna-<tag>-llama-<llama.cpp release>-win-x64.zip`
(`tools/package.ps1`) and publishes it, with a `.sha256` file holding the
zip's checksum, as a GitHub pre-release whose notes are
`.github/release-notes.md`. The zip shall hold, at its top level:
- the pinned llama.cpp release's Windows Vulkan zip, unpacked and
  unmodified, from a fresh download;
- `ggml-xdna.dll`, `npu.exe` from that build, and
  `bfp16_gemm.xclbin` from `kernels/bfp16_gemm/prebuilt`;
- `README.md`, `LICENSE`, `NOTICE`, and `LICENSES\` with llama.cpp's
  license (from the same tag's source; its zip carries only OpenMP's), the
  XRT license and notice, and the mlir-aie license;
- `SHA256SUMS.txt`, every other file's SHA-256 in `sha256sum -c` format.

Nothing else: none of the test programs the build copies next to llama.cpp.
Started by hand, or by a pull request that changes the packaging, the
workflow publishes nothing and keeps the zip as the run's artifact.

The zip is unsigned, like llama.cpp's own Windows builds.

**Verification (manual):** a dry run. Start the workflow by hand (or open a
pull request touching the packaging), download the artifact, unzip it into
an empty folder, and from there, with no `GGML_*` or `LLAMA_*` variable set:
- `npu llama-completion -m <Qwen3-4B Q4_K_M> -f <a 2,000+-token prompt>
  -n 24 --temp 0 -no-cnv`: prints `npu: prompts on the NPU`, and the reply
  is sensible;
- `npu llama-server -m <the same>`, one chat request of 2,000+ tokens: a
  sensible reply, and the server's log shows the NPU took pieces
  (`GGML_XDNA_TRACE=1` for this one);
- every line of `SHA256SUMS.txt` checks, and the file list matches the
  above.

---

### XDNA-DISPATCH-COST: Submission overhead stays under a millisecond — RETIRED 2026-09-30

Retired with the move of the research code out of the tree. It measured the
per-call cost that the first, one-matmul-at-a-time design depended on, with
`tools/bench-dispatch.cpp` (in the `research-archive` tag). Block claiming
doesn't depend on it, and the model-level speed numbers cover it. Measured
2026-09-13: ~0.1 ms a call, 0.02–0.04 ms batched.

---

## Below the traceability line

- The host reference matmul in `src/xdna-ref.cpp` is a placeholder for the NPU
  kernels. Its threading, blocking and dequant strategy are implementation
  detail; only XDNA-MUL-MAT-AGREES constrains it.
- The exact default values of `GGML_XDNA_MIN_BATCH` and `GGML_XDNA_MIN_MFLOP`
  are tuning, not contract. The requirements fix the mechanism, not the numbers.
  The same goes for how much memory XDNA-MEMORY-BUDGET keeps back (4 GB or a
  tenth of memory): a starting point, to be tuned from measurements.
- The backend's weight cache (step 1: raw bytes of each claimed weight, kept
  for tensors in buffers llama.cpp marks as weights, keyed on buffer, offset,
  type and shape) and the pinned host memory it reads activations into.
  Weight copies, raw and 8-bit, belong to the backend instance llama.cpp
  makes per context and are freed with it. That lifetime is what makes a
  key by location safe: a context never outlives its model. Anything that
  keeps copies longer (a disk cache, copies shared between contexts) must
  key them by content instead.
- `GGML_XDNA_TRACE=1`: a diagnostic that records the inputs of every op the
  backend declines and reports any of its own results no declined op was seen
  reading. It is the basis for deciding which results block claiming must
  write back.
- Tried and removed, 2026-10-02: asking the NPU driver for a quality of
  service when the add-on opens its NPU context (`xrt::hw_context` with
  `gops=50000` or `latency=1`). The driver received it (xrt-smi showed the
  values on the context), but on the NPU's "Default" power mode it made no
  difference: Qwen3-1.7B, 2,048-token prompt, 26 paired rounds, 0.99x and
  1.01x the speed with no request. xrt-smi's power mode made no
  difference either: switching between "Default" and "Performance" within
  one run, 24 rounds, the add-on ran at 0.995x its "Default" speed on
  "Performance" (quartiles 0.97–1.03x). Two earlier pairs of runs taken at
  different times had disagreed (+10% and none); the whole chip's speed
  drifts, so only the within-run test counts.
- Which llama.cpp release we pin.
