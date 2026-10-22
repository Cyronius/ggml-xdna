ggml-xdna puts an AMD Ryzen AI NPU to work on long prompts in Lemonade.
The GPU handles attention and writes the reply. This is an independent
community add-on, tested with Lemonade 2026.43.3 and the bundled b10944 backend.

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
Speed depends on the model. A runtime NPU failure can leave the affected
prompt finishing slowly on CPU; later requests use the GPU. The failure
check took about 63 seconds for that request.

The desktop walkthrough uses screenshots from the actual app. It shows
prompt entry and the answer, but isn't a real-time latency recording.

## Direct llama.cpp use

Still supported on Windows and Linux. The `npu` launcher is included in
both packages. See the [technical guide](https://github.com/Cyronius/ggml-xdna/blob/main/docs/technical-guide.md)
for commands, build instructions, accuracy checks, and standalone benchmarks.
