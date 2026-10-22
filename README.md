<p align="center">
  <a href="https://lemonade-server.ai/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/lemonade-logo-dark.png">
      <img src="docs/assets/lemonade-logo-light.png" alt="Lemonade" width="300">
    </picture>
  </a>
</p>

# ggml-xdna: use your NPU in Lemonade

ggml-xdna puts your AMD Ryzen AI NPU to work on long prompts. You chat in
Lemonade; your GPU still handles attention and writes the reply. It works
with local GGUF models, without compiling an NPU kernel for each model.

**[Download the Windows preview](https://github.com/Cyronius/ggml-xdna/releases/tag/lemonade-preview-20261021-r2)**
· [Setup guide](docs/lemonade.md)
· [Website](https://cyronius.github.io/ggml-xdna/)
· [Direct llama.cpp / Linux](docs/technical-guide.md)

This is an independent community add-on. The tested preview uses
Lemonade **2026.43.3** and the bundled **llama.cpp b10944** backend.
Keep that backend together: the DLL won't work with an arbitrary newer build.

## See it running

[![Actual Lemonade desktop chat using Qwen3-4B with NPU-assisted prompt processing](docs/results/lemonade-desktop-answer.png)](https://github.com/Cyronius/ggml-xdna/releases/download/lemonade-preview-20261021/lemonade-desktop-walkthrough.mp4)

[Watch the 19-second walkthrough](https://github.com/Cyronius/ggml-xdna/releases/download/lemonade-preview-20261021/lemonade-desktop-walkthrough.mp4)
· [NPU trace](docs/results/lemonade-desktop-trace.txt)
· [Verification report](docs/lemonade-validation.md)

That's the actual Lemonade app. It loaded Qwen3-4B, answered a 2,780-token
prompt, and logged 344 NPU calls. The video is a walkthrough made from
desktop screenshots, so don't use it to judge latency.

## Try it on Windows

You'll need Windows 11, an AMD Ryzen AI NPU with its driver, a Vulkan GPU
driver, [Lemonade 2026.43.3](https://github.com/lemonade-sdk/lemonade/releases/tag/v2026.43.3),
and a GGUF file already on disk. We tested a Ryzen AI 9 HX PRO 370 with
Radeon 890M graphics. Qwen3-4B Q4_K_M is a useful starting point if you want
to repeat that test. Lemonade, drivers, and model weights are separate downloads.

1. [Download the Windows preview](https://github.com/Cyronius/ggml-xdna/releases/tag/lemonade-preview-20261021-r2)
   and extract the ZIP. Keep the files together. There's a SHA-256 checksum
   beside the download and a file manifest inside it.
2. Quit the existing Lemonade tray server, then double-click
   **`Start-Lemonade.cmd`** in the extracted folder. It looks for Lemonade
   and asks you to find its server executable if needed. Choose your GGUF
   file when asked, and leave the launcher window open.
3. Open Lemonade App. In **Settings → Base URL**, enter
   **`http://127.0.0.1:13315`** and click **Save**. Select your imported model
   in the chat picker and try a long prompt.

The launcher keeps its configuration separate and uses the GPU if the NPU
can't start. To go back to your regular setup, stop this server with Ctrl+C,
restart your normal Lemonade server, and restore the app's previous Base URL
(normally `http://localhost:13305`).

The [setup guide](docs/lemonade.md) also covers terminal commands, portable
Lemonade, logs, and troubleshooting. If you're running from source, build
with `build.cmd` first.

## What the NPU does

| Work | Processor |
|---|---|
| Eligible matrix multiplies and nearby operations in large prompt chunks | NPU |
| Attention, small prompt chunks, and reply generation | Vulkan GPU |

The gain comes while the model reads your prompt, usually called *prefill*.
Reply generation stays on the GPU. Chunks below 1,024 tokens normally stay
there too, so a quick greeting may show no NPU activity. The first long
request can take longer while the NPU finishes converting its weight copies.

## What we measured in Lemonade

![Three paired Lemonade runs: GPU-only prefill took 6445, 6623, and 7656 milliseconds; NPU-assisted prefill took 5113, 5227, and 5334 milliseconds](docs/assets/lemonade-results.svg)

Qwen3-4B Q4_K_M read a 2,781-token prompt at a median **1.267×** the GPU-only
speed, or about **21% less prefill time**. That's three warm pairs on one
laptop in one sitting. Background load and thermals weren't controlled, and
we haven't shown a decode benefit or compared this with Lemonade's current
default GPU backends. Try your own model before assuming it'll be faster.

[Full responses and hashes](docs/results/lemonade-2026-10-21.json)
· [NPU trace](docs/results/lemonade-2026-10-21-npu-trace.txt)
· [Repeat the comparison](docs/lemonade.md#measure-your-model)
· [Standalone model benchmarks](docs/technical-guide.md#results)

## Before you use the preview

| Item | What we've checked |
|---|---|
| Windows 11, Ryzen AI 9 HX PRO 370 / Radeon 890M | Actual Lemonade desktop and API tests |
| Lemonade / backend | 2026.43.3 / bundled b10944 Vulkan build |
| Other Ryzen AI hardware, including Strix Halo | Not tested by this project |
| Linux | Builds and host tests pass; NPU hardware and Lemonade integration remain unverified |
| Models | Coverage depends on b10944 and the operations the add-on implements |
| Memory and output | Converted weight copies need extra RAM; block rounding changes numerical results |
| NPU unavailable at startup | Launcher chooses the GPU and logs the reason |
| NPU failure during a request | That prompt can finish slowly on CPU; later prompts use the GPU |

The request where we deliberately failed the NPU took about 63 seconds.
GPU-only and NPU-assisted runs answered the retrieval and arithmetic checks
correctly, but both returned an equation when asked for just an integer.
The [verification report](docs/lemonade-validation.md) includes those failures,
and the [technical guide](docs/technical-guide.md#limitations) covers the
memory, accuracy, and model limits in more detail.

## Help and direct llama.cpp use

- [Lemonade setup and troubleshooting](docs/lemonade.md)
- [Quality checks and benchmarks](docs/lemonade.md#measure-your-model)
- [Direct llama.cpp on Windows](docs/technical-guide.md#quick-start) and
  [Linux](docs/technical-guide.md#linux), both still supported
- [Build instructions, tuning, and architecture](docs/technical-guide.md)
- [Report a problem](https://github.com/Cyronius/ggml-xdna/issues/new) with
  your processor, drivers, Lemonade version, model, and launcher logs

MIT licensed. [License](LICENSE) · [Dependency and logo attribution](NOTICE).
The Lemonade logos are from its [official brand kit](https://github.com/lemonade-sdk/assets/tree/main/brand-kit).
