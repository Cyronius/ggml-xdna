# Windows Lemonade release verification

Checked 2026-10-21 on a Ryzen AI 9 HX PRO 370 with Radeon 890M.

## r2 preview: build, launcher, and recovery

The r2 preview adds `Start-Lemonade.cmd`, which looks for an installed
Lemonade server and opens file pickers for missing paths. Its terminal
options still work, including `-PrepareOnly` and `-GpuOnly`. File-picker
selection itself wasn't automated in these checks.

Main already contained the GPU recovery fix from PR #21. It is preserved
in this update. A fresh source archive of merge commit
`dd628e9` was built in an empty build directory using the verified official
b10944 archive and matching cached headers. No old objects or backend DLL
were reused. All **12 applicable CTest tests passed**, including real NPU
matmul, invalid-result injection, and submission-failure injection. The
no-driver test was excluded because this machine has the driver.
[CTest output](results/lemonade-r2-clean-build-ctest.txt).

The rebuilt DLL's SHA-256 is
`9ab00b2a12592b981410218e26e7211d0095073ce2c8de263f270fca0e62de0c`.
The final package's checksum and file verification receipt are attached to
the r2 GitHub release.

The CMD entry point prepared configuration and started the official server
from an extracted path containing spaces. Its saved settings were used by
Lemonade. With Qwen3-4B Q4_K_M and the rebuilt DLL, all three answer checks
passed; the arithmetic answer again returned an equation instead of just an
integer. Positive NPU call counts were recorded. Ctrl+C stopped the server.
[Responses](results/lemonade-preview-r2-api.json) and
[trace excerpts](results/lemonade-preview-r2-trace.txt).

For the recovery check, `GGML_XDNA_FAIL_AFTER=0` failed the first eligible
operation with Qwen3-1.7B Q4_0. Exactly one failure notice said the failed
work would finish on the GPU. All three answers were correct. First content
arrived in **4.28 seconds** on the affected request, then **4.52 and 2.40
seconds** on the following requests. The original preview's affected request
took 63.26 seconds while it finished on CPU. These are single checks, not a
controlled performance comparison. The CPU reference remains available if
the GPU cannot run a failed piece.
[Recovery responses](results/lemonade-r2-runtime-fallback-api.json) and
[trace](results/lemonade-r2-runtime-fallback-trace.txt).

The records below describe the original preview. Its slow CPU recovery
result is retained to show what changed; it does not describe r2's tested
recovery path.

## Original preview: build and package

The engine was built from a new `git archive` of
`3def7c63ec0cd3cb638d7984a2edf1263bcd5b1f`, in an empty source/build directory,
with freshly fetched b10944 headers and Windows Vulkan binaries. No existing
object files or local backend DLL were reused. MSVC reported version
19.44.35228.0.

All **11 applicable CTest tests passed**, including real NPU matmul and
memory-budget checks, invalid-result injection, startup rejection, host
operations, packing and instruction generation. The no-driver test was
excluded because this machine has the driver.
[Full CTest output](results/lemonade-clean-build-ctest.txt).

The clean DLL's SHA-256 is
`688de55fda612088192e7d605580966c13bb702d2ccd93eed41902e2a0cf76d8`.
The release ZIP was unpacked to a separate directory; every manifest entry
was verified, and the unpacked DLL matched that hash. Packaging changes add
the Lemonade tools, prompt fixture and documentation. They also verify the
official Windows llama.cpp archive's SHA-256 before extraction. A CMake test
path now uses the platform executable suffix for Linux CI.

## Original preview: actual Lemonade desktop app

The official **Lemonade 2026.43.3 MSI** was downloaded and checked against
GitHub's published asset digest. Its files were extracted without installing
or changing the system-wide service. The test used the release's actual
`LemonadeServer.exe` and Tauri `lemonade-app.exe`, rather than a substitute
chat client. The tray icon could not initialize in the test environment;
the official server continued in its supported headless mode.

The desktop app connected to the isolated server, selected the local
Qwen3-4B Q4_K_M model, loaded the packaged backend and displayed answers to
long prompts. Its own log panel showed NPU execution:

- A 2,773-token cold request reported **342 NPU calls**, then a 43-token
  answer. Prefill was 12.858 seconds, including unfinished weight conversion.
- A later prompt with a changed prefix processed 2,780 tokens and reported
  **344 NPU calls**, then a 46-token answer. Prefill was 6.327 seconds.
- Repeating the earlier prompt in a new chat reused the prefix cache and
  processed only one new prompt token. This cached request is not evidence
  of a full NPU prefill pass.

[Desktop answer screenshot](results/lemonade-desktop-answer.png) and
[server trace](results/lemonade-desktop-trace.txt).
The accompanying desktop walkthrough video is assembled from captured
windows showing prompt entry and the actual answer. It is a visual demo,
not a real-time latency recording. The ordinary GDI recorder returned black
frames and its output was rejected.

Desktop testing caught a real configuration error: saved `merge_args=false`
caused Lemonade to discard inherited batch/tensor-split arguments. The
launcher now saves `merge_args=true`. The effective options and actual
subprocess command were checked before accepting the desktop result.

## Original preview: quality and streaming

GPU-only and NPU-assisted requests went through the full Lemonade server's
streaming OpenAI API. Both retrieved an access code near the beginning and
end of roughly 2,800-token prompts. Both answered the arithmetic check
correctly, but returned `27 + 15 - 8 = 34` despite the requested integer-only
format. The original strict check therefore reports two formatting failures,
one for each configuration; this is retained in
[the full responses](results/lemonade-clean-package-api.json).

`tools/check-lemonade.py` now reports answer correctness and formatting
separately; `--strict-format` makes formatting failures fatal. Its relaxed
arithmetic check accepts only the correct integer or this exact correct
equation, rather than arbitrary text containing the answer. These three
tasks are functional checks, not a general model-quality evaluation.

A separate model-level comparison using the packaged `llama-perplexity`,
eight 512-token chunks and 2,048-token batches, measured **mean KL divergence
0.004834** and **97.745% same top token** against Vulkan. This is within the
[technical guide's](technical-guide.md#accuracy) 0.01 dense-model tolerance. The corpus repeats the repository's
prose fixture; the result is specific to that corpus.
[KL record and hashes](results/lemonade-clean-package-kl.json).

## Original preview: fallback

For startup rejection, `GGML_XDNA_KERNELS` deliberately named a nonexistent
file. The launcher detected that no XDNA device was offered and configured
`Vulkan0`. Three streamed answer checks then succeeded through the full
Lemonade server using Qwen3-1.7B Q4_0. The arithmetic response again had the
baseline formatting difference.
[Responses](results/lemonade-startup-fallback-api.json) and
[trace](results/lemonade-startup-fallback-trace.txt).
This simulates an unavailable NPU; the driver was not uninstalled.

The rejection test also caught an initial probe bug: matching `XDNA0:`
anywhere accepted a warning saying it was not offered. The launcher now
requires a device-list entry at the beginning of a line.

For runtime recovery, `GGML_XDNA_FAIL_AFTER=0` deliberately failed the first
eligible operation with Qwen3-1.7B Q4_0. Exactly one failure notice was logged.
All three streamed answer checks succeeded. The affected request took
**63.26 seconds to first content** while the pre-scheduled prompt work
finished on the CPU. The later GPU requests took **5.27 and 3.03 seconds**.
This verifies correct recovery, not fast recovery; a failed prompt can be
very slow. [Responses](results/lemonade-runtime-fallback-api.json) and
[trace](results/lemonade-runtime-fallback-trace.txt).

## Supported scope

This verifies a pinned **Windows integration**, using Lemonade 2026.43.3 and
llama.cpp b10944. It does not establish compatibility with Lemonade's newer
default llama.cpp binary, Linux NPU execution, or Strix Halo. The add-on's
extra memory use, numerical rounding, model-dependent performance and
runtime recovery limitations still apply. None of the timing figures in
this check is a controlled cross-engine ranking.
