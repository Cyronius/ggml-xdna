ggml-xdna accelerates long prompts for many GGUF models through Lemonade's
llama.cpp backend, using an AMD Ryzen AI NPU. The GPU handles attention and
writes the reply. No ONNX or Q4NX export or per-model NPU kernel compilation
is required. Coverage follows the bundled build and supported operations;
speedups vary by model.

This is an independent community add-on, tested with Lemonade 2026.43.3
and the bundled b10944 backend.

## Use it with Lemonade on Windows

1. Download the Windows ZIP below and extract it. Keep the files together.
2. Quit your existing Lemonade tray server and double-click `Start-Lemonade.cmd`.
   Choose Lemonade's server executable if asked, then your local GGUF model.
3. In Lemonade App, set **Settings > Base URL** to `http://127.0.0.1:13315`
   and save. Select the imported model and send a long prompt.

The launcher keeps its configuration separate and uses the GPU if the NPU
can't start. It doesn't include Lemonade, model weights, or AMD drivers.
The ZIP has an internal checksum manifest; its `.sha256` file is attached.

[Website and walkthrough](https://cyronius.github.io/ggml-xdna/)
? [Lemonade setup and troubleshooting](https://github.com/Cyronius/ggml-xdna/blob/main/docs/lemonade.md)
? [Test report](https://github.com/Cyronius/ggml-xdna/blob/main/docs/lemonade-validation.md)

## Before you use it

We tested Windows 11 on a Ryzen AI 9 HX PRO 370 with Radeon 890M graphics.
Other Ryzen AI hardware, including Strix Halo, hasn't been verified by this
project. Linux builds and host tests pass; Linux NPU hardware and Lemonade
integration remain unverified.

The NPU keeps extra converted weight copies in RAM and changes numerical
results through block rounding. Small prompt chunks stay on the GPU.
Speed depends on the model. If the NPU fails during a request, failed work
is retried on the GPU and later requests use the GPU. The CPU reference
remains available if the GPU cannot run a failed piece. This build includes
the GPU recovery fix already merged into main.

The desktop walkthrough uses screenshots from the actual app. It shows
prompt entry and the answer, but isn't a real-time latency recording.

## Direct llama.cpp use

Still supported on Windows and Linux. The `npu` launcher is included in
both packages. See the [technical guide](https://github.com/Cyronius/ggml-xdna/blob/main/docs/technical-guide.md)
for commands, build instructions, accuracy checks, and standalone benchmarks.
