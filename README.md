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
puts the add-on (`ggml-xdna.dll`), the NPU kernel (`bfp16_gemm.xclbin`) and
the launcher (`npu.exe`) next to it.

From that folder, put `npu` in front of the llama.cpp command:

```
npu llama-server
npu llama-cli -m model.gguf
```

With no model given, `npu` lists the models it finds and asks which one:

```
Models found:
   0  all of them: llama-server's router, which loads each model when it's asked for
   1  granite-4.1-3b-Q4_K_S                 2.0 GB  LM Studio: unsloth
   2  Qwen3.8-27B-UD-IQ3_S                 12.0 GB  LM Studio: unsloth
   3  gpt-oss-20b-MXFP4                    12.1 GB  Hugging Face: ggml-org
Pick a number (Enter: granite-4.1-3b-Q4_K_S; q: quit):
```

It looks in a `models` folder next to `npu.exe`, LM Studio's models folder,
llama.cpp's download folder and the Hugging Face download folder. Enter
picks the model you chose last time. "All of them" (llama-server only)
serves every model in the list and loads each when a request names it.

What `npu` does for you:
- points llama.cpp at the add-on next to it;
- checks the NPU can run, and if so adds `-dev XDNA0,Vulkan0`. If it can't,
  it adds nothing, says why in one line, and llama.cpp runs on the GPU;
- adds `-ub 2048 -b 2048` for llama-server, llama-cli and llama-completion
  when the NPU is on, so prompts reach it in bigger pieces (faster, see
  below). Pass `-ub 512` to keep llama.cpp's default.

Anything you give yourself (`-m`, `-dev`, `-ub`, `-b`) is kept, and the
rest of your command reaches llama.cpp exactly as typed. `npu server` works
for `npu llama-server` too. Put the folder on your `PATH` to run `npu` from
anywhere.

**Without the launcher,** set `GGML_BACKEND_PATH` to the add-on and name
the NPU yourself:

```
set GGML_BACKEND_PATH=C:\path\to\llama-b10944\ggml-xdna.dll
llama-server -m model.gguf -dev XDNA0,Vulkan0
```

Without `-dev XDNA0,...`, llama.cpp runs as if the add-on weren't there.

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
  in the background while the model loads, which takes a few seconds (4.4 s
  for Qwen3-4B). llama-server is usually ready before the first request; a
  prompt given on the command line waits for the rest.
  The copies are kept within the memory free once the model is loaded, less
  4 GB or a tenth of the machine's memory, whichever is larger. When they
  don't all fit, the first layers that do go to the NPU and the rest stay
  on the GPU, and the add-on says so:
  `xdna: NPU weight copies limited to 20.0 GB (...): the first 40 layers on
  the NPU, the rest on the GPU`. `GGML_XDNA_MAX_COPY_GB` sets the limit.
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

When the NPU can't run, `npu` says why and runs llama.cpp on the GPU:

```
npu: running on the GPU only: <reason>
```

Without the launcher, the add-on offers no XDNA0 device and logs
`xdna: not offering XDNA0: <reason>`. A command that names
`-dev XDNA0,Vulkan0` itself then stops with `invalid device: XDNA0`; use
`npu`, or `-dev Vulkan0`, until the reason is fixed.

| reason | what to do |
|---|---|
| no bfp16_gemm.xclbin next to ggml-xdna.dll | copy it there, or set `GGML_XDNA_KERNELS` to it |
| GGML_XDNA_KERNELS=... names no xclbin | fix the path |
| no NPU driver (xrt_coreutil.dll not found) | install the AMD NPU driver |
| the NPU driver's xrt_coreutil.dll lacks a function this backend uses | the driver is older or newer than the add-on was built for; report it |
| no NPU found (...) | the driver doesn't see an NPU |
| the NPU "..." hasn't been tested with this backend | a chip other than Ryzen AI 300; `GGML_XDNA_ANY_NPU=1` tries it |
| the NPU won't load ... | another program may be holding the whole NPU |
| the add-on didn't load | `ggml-xdna.dll` isn't next to `npu.exe`, or `GGML_BACKEND_PATH` points elsewhere |

**Is it doing anything?** `npu` prints `npu: prompts on the NPU` when it
turns it on. `set GGML_XDNA_TRACE=1` prints where the time goes every few
hundred multiplies. A prompt under 512 tokens never reaches the NPU.

**Reporting a problem:** open an issue with your chip, the NPU and GPU driver
versions, the model, the command, and the output with `GGML_XDNA_TRACE=1`.

## Settings

All optional, and none needed with `npu`. They're environment variables
(`set GGML_XDNA_TRACE=1` in cmd, `$env:GGML_XDNA_TRACE = "1"` in
PowerShell), read once when llama.cpp starts: set them before starting it.
An empty value counts as unset.

| variable | default | |
|---|---|---|
| `GGML_XDNA_KERNELS` | `bfp16_gemm.xclbin` next to the DLL | another xclbin |
| `GGML_XDNA_ANY_NPU` | 0 | 1 tries an NPU the add-on hasn't been tested on |
| `GGML_XDNA_MIN_BATCH` | 512 | the smallest piece of a prompt, in tokens, the NPU takes |
| `GGML_XDNA_MIN_MFLOP` | 256 | the smallest multiply the NPU takes, in millions of operations |
| `GGML_XDNA_COPY_AT_LOAD` | 1 | 0 builds the NPU's weight copies during the first prompt instead of while the model loads |
| `GGML_XDNA_MAX_COPY_GB` | memory free at load, less 4 GB or a tenth of memory | the most memory, in GB, the NPU's weight copies may take; 0 keeps the NPU out |
| `GGML_XDNA_BLOCKS` | 1 | 0 takes only the multiplies, not the steps between them |
| `GGML_XDNA_STREAMS` | 2 | how many parts a piece is split into, so the CPU and NPU overlap |
| `GGML_XDNA_N_THREADS` | all cores | CPU threads for the add-on's own work |
| `GGML_XDNA_TRACE` | 0 | 1 prints where the time goes |
| `GGML_XDNA_PINNED` | 1 | 0 reads from the GPU into ordinary memory instead of pinned memory: slower, for software Vulkan devices |
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
