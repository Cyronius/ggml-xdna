# NPU prefill for llama.cpp, on AMD XDNA2

The goal is to read prompts on the NPU in a Ryzen AI laptop and let llama.cpp
do everything else, with no fork of llama.cpp and no patched build.

Two pieces live here, and they are stages of one thing rather than two
projects:

- **`ggml-xdna`** ([src/](src/)) — a backend DLL that loads into a stock
  llama.cpp release. This is where the work is meant to end up.
- **`hybrid/`** — a prototype that uses llama.cpp as a library instead of
  plugging into it. It exists so the NPU side can be built and measured
  without also fighting ggml's scheduler. It is scaffolding, not a rival.

The NPU kernels themselves are not here. They come from
[OpenFlowLM-Next](https://github.com/Atomic-Germ/OpenFlowLM-Next) and are
consumed as built artifacts. This repo is the seam, and it builds standalone:
the XRT headers and the small `extern "C"` shim over them are vendored under
[`vendor/`](vendor/), so no other checkout has to be present.

## Where this stands

**The weight path works on real hardware.** Qwen3-1.7B weights, read from a
GGUF file, rearranged into the layout the NPU kernels expect, and dispatched
through the vendored driver, come back correct — inside the kernel's own error
budget, at 1.44 trillion operations per second.

**The handoff works.** A prompt read by our own code outside llama.cpp can be
pushed into a llama.cpp context, which then generates exactly what it would
have generated had it read the prompt itself. Same first token, same 24-token
continuation. Costs 2.3 ms for 58 MB of state.

**What is missing is one kernel: batched attention.** Everything else is done
or is routine porting. See [.claude/plans/hybrid-prefill-amd-design.md](.claude/plans/hybrid-prefill-amd-design.md).

**The backend half is an unfinished first attempt at the wrong granularity.**
It claims individual matrix multiplies. Everything else in a forward pass —
attention, the norms — stays on the CPU, and that is roughly half of prompt
reading. Measured in 2026-09, handing the NPU every matrix multiply for free
still capped prompt reading below what the integrated GPU does on its own.

That result is often quoted as "NPU prefill is dead." It is narrower than it
sounds. It rules out *a backend that claims matrix multiplies*. It says
nothing about a backend that claims a whole transformer block and fires one
fused dispatch for it, which is what AMD's own hybrid mode does and what
ggml's scheduler is perfectly capable of handing over — it groups neighbouring
operations assigned to the same backend and passes them across in one call.
Nobody has tried that here. Getting there needs the batched attention kernel
first, which is why `hybrid/` exists and why the backend is parked rather than
abandoned.

## How the current backend claims work

There is no phase handoff anywhere in this code. `ggml_backend_sched` picks,
for each node, the highest-priority backend that supports **both** the weight's
buffer type and the op (`ggml/src/ggml-backend.cpp`, `backend_from_buffer`). So:

- Our buffer type reports `is_host = true`, which makes the CPU backend accept
  it too (`ggml_backend_cpu_device_supports_buft` takes any host buft).
- We register as an ACCEL device, so llama.cpp puts our buffer type first in the
  CPU buffer-type list and the weights land in our memory.
- `supports_op` claims `MUL_MAT` only at batch >= `GGML_XDNA_MIN_BATCH`
  (default 32).

Prefill matmuls come to us. Decode's `n_tokens=1` matmuls fail the batch gate
and fall through to the CPU, reading the same bytes. One copy of the weights,
no cross-backend transfers.

`offload_op` is deliberately not implemented — that path copies the weight into
our buffer per op, which is how OllamaAMDNPU ended up at 0.65 tok/s.

## Build

Needs MSVC (BuildTools 2022) and CMake. `tools/fetch-llama.ps1` downloads the
pinned llama.cpp release plus matching headers and generates import libs from
the release DLLs.

```
powershell -File tools\fetch-llama.ps1
build.cmd
```

The DLL is copied next to the llama.cpp binaries so `ggml-base.dll` resolves.

Targets that drive the NPU directly (`bench-dispatch`, `test-npu-gemm`)
additionally need an XRT import lib, which XRT on Windows does not ship.
`tools/gen-xrt-implib.ps1` reconstructs one from the export table of the
driver's own `xrt_coreutil.dll`. Without it those targets are skipped and
everything else still builds.

```
powershell -File tools\gen-xrt-implib.ps1
```

## Run the backend

```
set GGML_BACKEND_PATH=C:\code\npu-prefill-engine\third_party\llama-b10944\ggml-xdna.dll
third_party\llama-b10944\llama-cli.exe -m <model.gguf> -dev XDNA0,Vulkan0 -fa on -b 2048 -ub 2048
```

The NPU kernel, `bfp16_gemm.xclbin`, sits next to `ggml-xdna.dll` (the
build copies it there). If the NPU can't run it (no NPU, a chip the backend
hasn't been tested on, a missing file), XDNA0 isn't offered and one log line
says why. llama.cpp then runs on the GPU as usual. If the NPU fails in the
middle of a run, the backend finishes that step on the CPU, logs it, and
hands everything to the GPU after that.

`-dev XDNA0,Vulkan0` opts in: the NPU takes the big prompt work (weight
matmuls on long prompts, and the norms, rotary and adds between them), the
GPU keeps the model's weights and everything else, including every reply
token. Without `-dev` naming XDNA0, llama.cpp runs as if the backend weren't
there. `-ub 2048` gives the backend chunks big enough to claim.

The one xclbin serves every model: the backend makes each matrix size's NPU
instructions itself, so no per-model kernel builds are needed.

| env | default | |
|---|---|---|
| `GGML_XDNA_KERNELS` | `bfp16_gemm.xclbin` next to the DLL | another xclbin, or a directory holding `final.xclbin` |
| `GGML_XDNA_ANY_NPU` | 0 | 1 tries an NPU the backend hasn't been tested on (tested: Strix Point) |
| `GGML_XDNA_HOST_ONLY` | 0 | 1 runs claimed matmuls on a CPU reference, never the NPU: for tests without an NPU |
| `GGML_XDNA_MIN_BATCH` | 1024 | prompt tokens at or above which work is claimed |
| `GGML_XDNA_MIN_MFLOP` | 256 | smallest matmul claimed, in MFLOP |
| `GGML_XDNA_BLOCKS` | 1 | claim whole blocks (0: matmuls only) |
| `GGML_XDNA_STREAMS` | 2 | row streams per block, so host and NPU work overlap |
| `GGML_XDNA_TRACE` | 0 | 1 prints where the time goes |
| `GGML_SCHED_DEBUG` | 0 | set to 2 to print per-node backend assignments |

## Run the prototype

All of these need a GGUF model and most need the GPU, so run them from the
release directory where the DLLs live.

```
third_party\llama-b10944\kv-handoff.exe   <model.gguf> [--repeat 34]
third_party\llama-b10944\ref-handoff.exe  <qwen3.gguf> [--threads N]
third_party\llama-b10944\test-q4-pack.exe <model.gguf> [--dump <dir>]
third_party\llama-b10944\test-npu-gemm.exe <model.gguf> --build <kernel-build-dir>
third_party\llama-b10944\gpu-prefill.exe  <model.gguf> --lens 256,512,1024,2048
```

The first four have a pass/fail and check the requirements in
[specs/hybrid-prefill/spec.md](specs/hybrid-prefill/spec.md). `gpu-prefill` has
no pass/fail — it is a measurement, and that spec says how to read it.

## Measuring anything here

Two traps on this machine, both of which have already produced a wrong number
that had to be withdrawn:

- **Prompt-reading speed on the GPU cannot be timed once.** A single run in a
  fresh process can be off by more than ten times, because some launches
  compile shaders first and the runs after that are still slow while clocks
  ramp. Take the median of many runs in one process and throw the first
  several away. `gpu-prefill.exe` does this.
- **NPU dispatch timing needs at least 30 iterations.** At ten, warm-up still
  moved the best time by 60%.

Generation speed does not need either caveat; it was steady from the first
reading.
