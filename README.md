# ggml-xdna: llama.cpp prompt reading on the AMD Ryzen AI NPU

ggml-xdna is an add-on for llama.cpp that reads prompts on the NPU of an
AMD Ryzen AI 300 laptop, next to the integrated GPU. It loads into an
unmodified llama.cpp Windows build. The GPU keeps doing everything else,
including writing the reply.

## Requirements

- Windows 11
- A Ryzen AI 300 series processor (Strix Point). Tested on the Ryzen AI 9
  HX 370 with Radeon 890M graphics. Other chips are turned down at start
  (see [Troubleshooting](#troubleshooting)).
- The AMD NPU driver. Tested with 32.0.20102.3930.
- A GPU driver with Vulkan. Tested with Radeon driver 32.0.31041.1004.
- llama.cpp release **b10944**, Windows Vulkan build
  (`llama-b10944-bin-win-vulkan-x64.zip`). The add-on is built against that
  release and must run with it.

## Quick start

There's no release zip yet. [Build from source](#building-from-source):
`build.cmd` downloads llama.cpp b10944 into `third_party\llama-b10944` and
puts `ggml-xdna.dll` and the NPU kernel, `bfp16_gemm.xclbin`, next to it.

Then point llama.cpp at the add-on and name the NPU first in the device list:

```
set GGML_BACKEND_PATH=C:\path\to\llama-b10944\ggml-xdna.dll
llama-server -m model.gguf -dev XDNA0,Vulkan0
```

`-dev XDNA0,Vulkan0` is what turns it on. Without it, llama.cpp runs exactly
as if the add-on weren't there.

For longer prompts, `-ub 2048` is faster (below): llama.cpp then hands over
the prompt in bigger pieces. `-b` must be at least `-ub`; its default, 2048,
is.

## Results

Qwen3-1.7B, Q4_0. Prompt reading speed with the add-on, against the GPU
alone at the same settings. Measured 2026-09-30.

| prompt tokens | default settings (`-ub 512`) | `-ub 2048` |
|---|---|---|
| 512 | 1.07x | (one piece either way; not measured) |
| 1,024 | 1.00x | 1.16x |
| 2,048 | 0.98x | 1.13x |
| 4,096 | 1.02x | 1.18x |

At default settings the NPU about ties the GPU. With 2,048-token pieces it
reads prompts 13–18% faster. The GPU alone read 512 tokens at about 1,700
tokens a second.

**How it was measured:** `llama-bench -r 3`, run alternately with
`-dev Vulkan0` and `-dev XDNA0,Vulkan0`, five rounds each; the table gives
the median of the five per-round ratios. The machine was otherwise idle.
Single timings on this machine vary by more than the differences above, so
compare only paired, repeated runs like these.

## Accuracy

The NPU works in 8-bit blocks, so its results differ from the GPU's by
rounding. Measured with `llama-perplexity --kl-divergence` against the GPU
alone, on Qwen3-1.7B, Qwen3-4B and Qwen2.5-1.5B:

- mean KL divergence 0.004–0.008;
- the same most likely next token about 95% of the time.

The GPU agrees with itself across llama.cpp builds about as well. Short
greedy replies usually come out word for word the same.

The same holds with llama-server's parallel slots, cached conversations,
contexts of 32,768 tokens, and two models served at once by its router.

## Limitations

- **Windows only.** Linux is planned for later.
- **Ryzen AI 300 only.** `GGML_XDNA_ANY_NPU=1` tries another NPU, untested.
- **Prompt reading only.** Replies are written one token at a time, and that
  stays on the GPU.
- **Pieces under 512 tokens stay on the GPU**, so short prompts don't use
  the NPU.
- **Memory:** the NPU keeps its own 8-bit copy of each weight it uses,
  about 1.1 GB per billion parameters, on top of llama.cpp's. It's built
  during the first prompt, which takes a few seconds (4.4 s for Qwen3-4B).
- **Some layers stay on the GPU:** mixture-of-experts layers, and a few
  variants the NPU side doesn't implement yet (bias adds, some rotary
  settings). Output is still correct; less of the work moves.
- If the NPU fails during a run, the add-on finishes that step on the CPU
  and hands everything to the GPU from then on. The rest of that prompt is
  slow.

## How it works

```mermaid
flowchart LR
    S[llama.cpp scheduler] -->|big prompt pieces:<br/>weight multiplies and the<br/>norms, rotary and adds between| X[XDNA0: the NPU]
    S -->|everything else:<br/>attention, short prompts,<br/>every reply token| V[Vulkan0: the GPU]
    X -. reads and writes .-> M[(GPU memory:<br/>weights and activations)]
    V --- M
```

The add-on registers a device, XDNA0, that shares the GPU's memory. llama.cpp
keeps every weight in GPU memory as usual, so work can move between the two
without copies. The add-on accepts only large prompt work: a model's weight
multiplies on pieces of 512 tokens or more, and the small steps between them,
so each hand-over covers most of a transformer block.

On the NPU, one kernel program serves every model: the add-on makes the NPU
instructions for each matrix size itself, so there's nothing to build per
model. The kernel works in bfp16, blocks of eight 8-bit values sharing one
exponent. Each weight is converted once, on first use. While the NPU runs one
part of the prompt, the CPU prepares the next.

## Troubleshooting

When the add-on can't use the NPU, it offers no XDNA0 device and logs one
line:

```
xdna: not offering XDNA0: <reason>
```

A command that names `-dev XDNA0,Vulkan0` then stops with
`invalid device: XDNA0`. Until the reason is fixed, use `-dev Vulkan0`, or
leave `-dev` out.

| reason | what to do |
|---|---|
| no bfp16_gemm.xclbin next to ggml-xdna.dll | copy it there, or set `GGML_XDNA_KERNELS` to it |
| GGML_XDNA_KERNELS=... names no xclbin | fix the path |
| no NPU driver (xrt_coreutil.dll not found) | install the AMD NPU driver |
| the NPU driver's xrt_coreutil.dll lacks a function this backend uses | the driver is older or newer than the add-on was built for; report it |
| no NPU found (...) | the driver doesn't see an NPU |
| the NPU "..." hasn't been tested with this backend | a chip other than Ryzen AI 300; `GGML_XDNA_ANY_NPU=1` tries it |
| the NPU won't load ... | another program may be holding the whole NPU |

If llama.cpp says `invalid device: XDNA0` and no `xdna:` line appears,
`GGML_BACKEND_PATH` doesn't point at `ggml-xdna.dll`.

**Is it doing anything?** `GGML_XDNA_TRACE=1` prints where the time goes
every few hundred multiplies. With default settings, a prompt under 512
tokens never reaches the NPU.

**Reporting a problem:** open an issue with your chip, the NPU and GPU driver
versions, the model, the command, and the output with `GGML_XDNA_TRACE=1`.

## Settings

All optional. An empty value counts as unset.

| variable | default | |
|---|---|---|
| `GGML_XDNA_KERNELS` | `bfp16_gemm.xclbin` next to the DLL | another xclbin |
| `GGML_XDNA_ANY_NPU` | 0 | 1 tries an NPU the add-on hasn't been tested on |
| `GGML_XDNA_MIN_BATCH` | 512 | the smallest piece of a prompt, in tokens, the NPU takes |
| `GGML_XDNA_MIN_MFLOP` | 256 | the smallest multiply the NPU takes, in millions of operations |
| `GGML_XDNA_BLOCKS` | 1 | 0 takes only the multiplies, not the steps between them |
| `GGML_XDNA_STREAMS` | 2 | how many parts a piece is split into, so the CPU and NPU overlap |
| `GGML_XDNA_N_THREADS` | all cores | CPU threads for the add-on's own work |
| `GGML_XDNA_TRACE` | 0 | 1 prints where the time goes |
| `GGML_XDNA_HOST_ONLY` | 0 | 1 runs the NPU's share on the CPU instead: for tests without an NPU |

## Building from source

Needs Visual Studio 2022 or its Build Tools, with the C++ tools (they bring
CMake and Ninja). Then, from the repository:

```
build.cmd
```

It downloads llama.cpp b10944 and its matching headers into `third_party\`,
builds the add-on and its tests, copies the DLL and the kernel next to the
llama.cpp binaries, and runs the tests. The NPU tests run only where the NPU
driver is installed; the others need a Vulkan GPU. `build.cmd notest` skips
the tests.

The build needs no NPU driver: the list of driver functions the add-on uses
is in [vendor/xrt-implib](vendor/xrt-implib/xrt_coreutil.def).
`tools\check-xrt-driver.ps1` checks an installed driver against it.

The NPU kernel ships prebuilt
([kernels/bfp16_gemm/prebuilt](kernels/bfp16_gemm/prebuilt)). Rebuilding it
needs AMD's IRON toolchain; see [kernels/bfp16_gemm/build.ps1](kernels/bfp16_gemm/build.ps1).

What the add-on must do, and how each part is checked, is in
[specs/xdna-backend/spec.md](specs/xdna-backend/spec.md).

## License

MIT ([LICENSE](LICENSE)). The vendored XRT headers and the NPU kernel keep
their own licenses; see [NOTICE](NOTICE).
