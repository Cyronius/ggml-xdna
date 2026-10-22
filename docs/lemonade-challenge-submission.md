# Lemonade Developer Challenge submission draft

## Project title

Zest: open-source NPU prefill for GGUF models in Lemonade

## Project link

https://github.com/Cyronius/zest

## Description

Zest adds AMD Ryzen AI NPU acceleration to llama.cpp's GGUF inference
path. Large prompt chunks send eligible matrix multiplies and nearby
operations to the NPU. Attention and token generation stay on the Vulkan
GPU. The backend code is MIT licensed; vendored components retain their
licenses as documented in NOTICE.

I integrated it with Lemonade 2026.43.3 using Lemonade's custom backend
binary setting and an unmodified llama.cpp b10944 Vulkan release. Neither
Lemonade nor llama.cpp requires a source patch. A chat request enters
Lemonade's OpenAI-compatible API, reaches llama-server with the add-on,
and returns through Lemonade. Runtime traces confirm NPU execution.

AMD already offers hybrid NPU/GPU inference. This project's contribution is
an open-source add-on for the llama.cpp/GGUF path, with no per-model NPU
kernel compilation. Its model coverage follows the pinned llama.cpp build
and the operations the add-on implements.

The repository includes setup instructions and an API comparison script
that holds the model and llama.cpp version constant, disables prompt cache
reuse, warms each configuration, and alternates GPU-only and NPU-assisted
runs. Existing standalone benchmarks and accuracy checks are documented
separately in the README.

On a Ryzen AI 9 HX PRO 370 with Radeon 890M, three paired runs through
Lemonade on Qwen3-4B Q4_K_M with a 2,781-token prompt gave a median 1.267x
prefill speed ratio (about 21% less prefill time). These are warm,
server-reported timings on one workload. Background load and thermals were
not controlled; the runs establish no decode advantage. Full measured
responses, artifact hashes and NPU trace excerpts accompany the result.

## Limitations and further work

The integration was verified on Windows with a Ryzen AI 9 HX PRO 370 and
Radeon 890M. It uses the add-on's required b10944 build rather than Lemonade's
newer default. The NPU keeps additional converted weight copies and changes
numerical results through block rounding. Small prompt chunks remain on the
GPU; speedups depend on the model, quantization, prompt length and hardware.
Linux integration and Strix Halo performance have not been verified by this
project. A larger Strix Halo GPU may change the balance substantially.

## Supporting material

- [Windows preview package and desktop walkthrough](https://github.com/Cyronius/zest/releases/tag/v0.2.0-rc2)
- [Lemonade setup](https://github.com/Cyronius/zest/blob/main/docs/lemonade.md)
- [Clean build, actual desktop app, quality and fallback verification](https://github.com/Cyronius/zest/blob/main/docs/lemonade-validation.md)
- [Repeated API comparison script](https://github.com/Cyronius/zest/blob/main/tools/bench-lemonade.ps1)
- [Measurement record](https://github.com/Cyronius/zest/blob/main/docs/results/lemonade-2026-10-21.json)
- [README](https://github.com/Cyronius/zest): architecture, standalone performance, accuracy and limitations

This remains submission copy for the owner's review. The challenge page
does not confirm remaining laptop availability. The desktop walkthrough
uses captured screenshots rather than a real-time latency recording.
