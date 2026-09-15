# ggml-xdna — NPU prefill backend for llama.cpp

A ggml backend that claims large-batch matmuls and runs them on the AMD XDNA2
NPU, leaving decode to the CPU. It builds as a standalone DLL and loads into a
stock llama.cpp release via `GGML_BACKEND_PATH` — no fork, no in-tree patch.

The kernels come from [OpenFlowLM-Next](https://github.com/Cyronius/OpenFlowLM-Next)
(`open_kernels/`). This repo is only the seam, and it builds standalone: the
`extern "C"` XRT shim and the XRT headers it needs are vendored under
[`vendor/`](vendor/), so no other checkout has to be present.

Current state is in the Status section below.

## How the prefill/decode split works

There is no phase handoff anywhere in this code. `ggml_backend_sched` picks, for
each node, the highest-priority backend that supports **both** the weight's
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

`bench-dispatch` additionally needs an XRT import lib, which XRT on Windows does
not ship. `tools/gen-xrt-implib.ps1` reconstructs one from the export table of
the driver's own `xrt_coreutil.dll`. Without it that one target is skipped and
everything else still builds.

```
powershell -File tools\gen-xrt-implib.ps1
```

## Run

```
set GGML_BACKEND_PATH=C:\code\ggml-xdna\third_party\llama-b10944\ggml-xdna.dll
third_party\llama-b10944\llama-cli.exe -m <model.gguf> -dev none -p "..."
```

`-dev none` keeps Vulkan from taking the weights first. Vulkan's `supports_buft`
only accepts its own buffers, so weights in ours are invisible to it — sharing
them with the iGPU would mean a second copy.

| env | default | |
|---|---|---|
| `GGML_XDNA_MIN_BATCH` | 32 | batch size at or above which we claim a matmul |
| `GGML_XDNA_N_THREADS` | hw concurrency | threads for the host reference matmul |
| `GGML_SCHED_DEBUG` | 0 | set to 2 to print per-node backend assignments |

## Status

The seam works and is the part worth keeping. The matmul is still a host
reference implementation (dequantise, then dot) - it exists to prove
scheduling and weight placement, and it is slow on purpose.

The NPU kernel work lives in `open_kernels/designs/gemm_q4/`. The blocked
batched q4 GEMM runs on all 32 compute tiles, eats Q4_K chunks as stored, and
streams a host-quantised activation table; the best configuration measured so
far carries 40 tokens per weight pass at 80.9 tok/ms. No kernel is wired into
this backend yet.
