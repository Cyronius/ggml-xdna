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
- **Memory:** the NPU keeps its own 8-bit copy of each weight it uses, about
  1.1 GB per billion parameters, on top of llama.cpp's. The add-on keeps
  these copies within the memory that's free; layers that don't fit stay on
  the GPU.
- **A pre-release.** The long server test isn't done yet.

**To use it:** unzip, open a terminal in the folder, and put `npu` in front
of the llama.cpp command:

```
npu llama-server -m model.gguf
```

If the NPU can't run, `npu` says why in one line and llama.cpp runs on the
GPU as usual. The zip isn't signed, so Windows may warn the first time you
run it ("More info", then "Run anyway"). `SHA256SUMS.txt` in the zip, and
the `.sha256` file next to it here, hold the checksums.

**Tested models:** TESTED-MODELS-TABLE

The README has the speed and accuracy measurements, the settings, and what
to do when it turns itself off.
