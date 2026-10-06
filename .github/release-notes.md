An add-on for llama.cpp that reads prompts on the NPU of an AMD Ryzen AI
laptop, next to the integrated GPU. This zip is llama.cpp's own Windows
Vulkan release (b10944), unmodified, with the add-on and its launcher next
to it.

**Before you try it:**
- **Windows 11 only.**
- **Ryzen AI chips only, and tested only on Strix Point** (Ryzen AI 9 HX
  370). Other Ryzen AI chips whose NPU passes the add-on's test at start are
  let in, untested.
- **Prompt reading only.** Replies are written on the GPU, as before.
- **Not every model gets faster.** Most do, by up to 1.9x. Mixture-of-experts
  models and models narrower than 2,048 come out slower; the add-on says so
  when it starts one, and runs it anyway.
- **Memory:** the NPU keeps its own 8-bit copy of each weight it uses, about
  1.1 GB per billion parameters, on top of llama.cpp's. The add-on keeps
  these copies within the memory that's free, and at most 20 GB; layers
  that don't fit stay on the GPU.

**To use it:** unzip, open a terminal in the folder, and put `npu` in front
of the llama.cpp command:

```
npu llama-server -m model.gguf
```

If the NPU can't run, `npu` says why in one line and llama.cpp runs on the
GPU as usual. The zip isn't signed, so Windows may warn the first time you
run it ("More info", then "Run anyway"). `SHA256SUMS.txt` in the zip, and
the `.sha256` file next to it here, hold the checksums.

**Tested models** (prompt reading in 2,048-token chunks, against the GPU
alone; KL divergence is how far the answers drift, where under 0.01 is about
as close as the GPU is to itself across llama.cpp builds):

| model | speed with the add-on | KL divergence |
|---|---|---|
| Qwen3.8-27B UD-IQ3_S | 1.88x | (reply matched) |
| LFM2.5 2.6B Q8_0 | 1.46x | 0.0045 |
| Qwen3-4B Q4_K_M | 1.33x | 0.004–0.008 |
| LFM2.5 1.2B Q8_0 | 1.31x | 0.0034 |
| Granite 4.1 3B Q4_K_S | 1.30x | 0.0073 |
| Gemma 4 E4B UD-Q4_K_XL | 1.01x | 0.0043 |
| Qwen3-1.7B Q4_0 | 0.97x | 0.004–0.008 |
| LFM2.5 8B-A1B UD-Q4_K_S (mixture of experts) | 0.97x | 0.020 |
| Ornith 35B / Qwen3.6 35B-A3B (mixture of experts) | 0.92x | 0.011 |
| LFM2.5-350M Q8_0 | 0.89x | |
| Qwen2.5-1.5B Q4_0 | 0.77x | 0.004–0.008 |

Speeds on models under about 3B move 8–14% from run to run. `bench.ps1` in
the zip reruns these measurements on your own models.

**Known issues:**
- If the NPU fails or its results come back damaged mid-prompt, the rest of
  that prompt runs on the CPU and can take many minutes (#13). The default
  memory limit keeps this from happening in normal use.
- Some steps Gemma, Granite, Phi-4 and Qwen2.5 use still run on the GPU,
  which is why Gemma only breaks even (#14).

The README has the full measurements, the settings, and what to do when it
turns itself off.
