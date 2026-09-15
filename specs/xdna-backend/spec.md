# XDNA backend — requirements

Canonical spec for the ggml backend that offloads prefill matmuls to the AMD
XDNA2 NPU. Home repo: `ggml-xdna`.

Kernels live in OpenFlowLM-Next (`open_kernels/`); this spec covers the seam
only — how the backend presents itself to `ggml_backend_sched`, and what its
matmul must produce. Requirements about the kernels themselves belong there.

Tests: `specs/xdna-backend/tests/` is empty by design — the test binaries are
built by the top-level `CMakeLists.txt` from `tests/`, because they must link
the same import libs as the backend. Run them with `GGML_BACKEND_PATH` set:

```
build.cmd
set GGML_BACKEND_PATH=...\third_party\llama-b10944\ggml-xdna.dll
third_party\llama-b10944\test-dispatch-gate.exe
third_party\llama-b10944\test-mul-mat.exe
```

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

### XDNA-HOST-BUFT: Weights stay readable by the CPU
**Applies to:** ggml-xdna
**Test category:** unit

The backend's buffer type shall report `is_host`, and the device shall register
as `GGML_BACKEND_DEVICE_TYPE_ACCEL`.

Together these are what keep a single copy of the weights: ACCEL puts our buffer
type first in llama.cpp's CPU buffer-type list so the weights land in our
memory, and `is_host` makes `ggml_backend_cpu_device_supports_buft` accept that
same buffer, so decode reads it in place instead of copying.

**Acceptance criteria:**
- `ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL`
- `ggml_backend_buft_is_host(ggml_backend_dev_buffer_type(dev))` is true

---

### XDNA-BATCH-GATE: Only large batches are claimed
**Applies to:** ggml-xdna
**Test category:** unit

`supports_op` shall return false for `GGML_OP_MUL_MAT` whose batch dimension is
below `GGML_XDNA_MIN_BATCH` (default 32), and true above it for supported quant
types. Batch is `ne[1]` for `MUL_MAT` and `ne[2]` for `MUL_MAT_ID`, matching the
Vulkan backend's convention.

This is the whole prefill/decode split: decode's `n_tokens=1` matmuls fail the
gate and the scheduler falls through to the CPU.

**Acceptance criteria:**
- batch 1 → not claimed
- batch 31 → not claimed
- batch 512, q4_K, 2048x2048 → claimed
- a quant type we cannot dequantise (e.g. `IQ2_XXS`) → not claimed at any batch

---

### XDNA-WORK-FLOOR: Small matmuls stay on the CPU
**Applies to:** ggml-xdna
**Test category:** unit

`supports_op` shall also return false when the matmul is below
`GGML_XDNA_MIN_MFLOP` (default 256) MFLOP, computed as
`2 * src0->ne[0] * src0->ne[1] * batch`.

A dispatch costs a few milliseconds regardless of size, so a small matmul is
cheaper left where it is. Without this the scheduler claimed the per-layer
`ssm_alpha` / `ssm_beta` projections (36 KB weights), adding 80 dispatches per
prefill graph for a few MFLOP of work.

**Acceptance criteria:**
- q4_K 512x64 at batch 512 (34 MFLOP) → not claimed
- q4_K 2048x2048 at batch 512 (4.3 GFLOP) → claimed

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
of 1.5, and shall have cosine similarity above 0.9999 with that reference.

Bit equality against the CPU backend is the wrong bar: ggml's CPU kernel
quantises the activation to int8 and we do not, so the two disagree by roughly
the activation quantisation error (~7e-3 NRMSE on q4_K) with the CPU being the
less accurate of the two.

**Acceptance criteria:**
- f32, f16, q8_0, q4_0, q4_K, q6_K, at shapes from 256x128x64 up to 2048x512x512
- each case: `nrmse_xdna <= 1.5*nrmse_cpu + 1e-6` and `cos_xdna > 0.9999`

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
- Which llama.cpp release we pin.
