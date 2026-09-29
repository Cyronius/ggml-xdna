"""Stage 0b: run GGUF-derived Q4_K weights through the real bpt2d1_m40 xclbin.

Part 1/2 of q4k_xform.py validated the chunk layout against the kernel's own
pool file and showed the encoder round-trips through my decoder. That does not
prove the encoder is what the *kernel* expects - encoder and decoder could share
a misunderstanding. This closes that loop on hardware.

Two comparisons:
  vs the bf16-narrowed fp64 model -> validates the encoder (expect ~1.6e-3,
     the kernel's own error)
  vs the exact GGUF fp64 model    -> total cost of the transform end to end

Run: python stage0b.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

SP = Path(__file__).parent
GEMM = Path(r"C:\code\openflowlm-next\open_kernels\designs\gemm_q4")
EXE = Path(r"C:\code\openflowlm-next\open_kernels\harness\out\run_kernel.exe")
GGUF = Path(r"C:\Users\josha\.flm\models\Qwen3.6-27B-A2.8B-open\qwen36-27b-a2.8b-mtp-Q4KM.gguf")
sys.path.insert(0, r"C:\code\npu-prefill-engine\third_party\src-tmp\llama.cpp-b10944\gguf-py")
sys.path.insert(0, str(GEMM))

N, K, M = 8192, 2048, 40
CORES, BPC, BPT, NKT, RS, TILE = 32, 4, 2, 8, 2, 4736
BUILD = "build_bpt2d1_m40"

# reuse the validated encoder/reference helpers
src = (SP / "q4k_xform.py").read_text()
exec(src.split('print("=== Part 1')[0])

print("=== building chunks from a real GGUF Q4_K tensor ===")
import gguf
rd = gguf.GGUFReader(str(GGUF), "r")
t = next(t for t in rd.tensors
         if t.tensor_type.name == "Q4_K" and tuple(int(s) for s in t.shape) == (K, N))
raw = np.asarray(t.data).reshape(N, -1)
d, dmin, sc, mn, q = unpack_q4k(raw.reshape(-1), N, K)
print(f"  {t.name} {N}x{K}")

rng = np.random.default_rng(0)
x_bf16 = np.asarray(rng.standard_normal(K), dtype=bfloat16)
x = x_bf16.astype(np.float64)
exact = ref_from_fields(d, dmin, sc, mn, q, x)
narrowed = ref_from_fields(d, dmin, sc, mn, q, x, cast=bfloat16)

pool = encode_chunks(d, dmin, sc, mn, q)                      # [128][16][4736]

# make_test.py's repack: [core][tile][band][kt][part] -> [core][tile][kt][band][part],
# then column-interleave one DMA element per core of each column
g = pool.reshape(-1).reshape(CORES, BPC // BPT, BPT, NKT, RS, TILE)
g = g.transpose(0, 1, 3, 2, 4, 5)
g = np.ascontiguousarray(g).reshape(CORES, -1, TILE)
g = g.reshape(8, CORES // 8, -1, TILE).transpose(0, 2, 1, 3)
w_path = GEMM / "w_gguf_blkt2c32.bin"
np.ascontiguousarray(g).reshape(-1).tofile(w_path)
print(f"  wrote {w_path.name} ({w_path.stat().st_size} B)")

# prep_host writes to the shared x_preq_m{M}.bin, so save and restore it
x_src = GEMM / "x_gguf_src.bin"
x_bf16.tofile(x_src)
shared = GEMM / f"x_preq_m{M}.bin"
backup = SP / f"x_preq_m{M}.bin.bak"
shutil.copy2(shared, backup)
try:
    import prep_host
    prep_host.build(M, x_src)
    x_path = GEMM / "x_preq_gguf_m40.bin"
    shutil.copy2(shared, x_path)
finally:
    shutil.copy2(backup, shared)
    print(f"  restored {shared.name}")

y_path = GEMM / "y_gguf_m40.bin"
y_path.unlink(missing_ok=True)
w = "C:/code/openflowlm-next/open_kernels/designs/gemm_q4"
cfg = GEMM / "run_gguf_m40.cfg"
cfg.write_text(
    f"""device
xclbin X {w}/{BUILD}/final.xclbin
kernelx k X {w}/{BUILD}/insts.bin
buf w {w_path.stat().st_size} {w}/{w_path.name}
buf x {x_path.stat().st_size} {w}/{x_path.name}
buf y {4 * N * M}
run k w x y
run k w x y
dump y {w}/{y_path.name} {4 * N * M}
""", encoding="utf-8", newline="\n")

print(f"\n=== running {BUILD} on the NPU ===")
p = subprocess.run([str(EXE), cfg.name], cwd=GEMM,
                   capture_output=True, text=True, timeout=600)
for line in p.stdout.splitlines():
    if "state" in line or "DONE" in line or "error" in line.lower():
        print("  " + line.strip())
if not y_path.exists():
    sys.exit("no y dumped:\n" + p.stdout[-2000:])

n_tiles = BPC // BPT
got = (np.fromfile(y_path, np.float32).astype(np.float64)
       .reshape(8, n_tiles, CORES // 8, BPT, M, 64)
       .transpose(4, 0, 2, 1, 3, 5)
       .reshape(M, N))


def stats(a, b):
    rel = np.abs(a - b).max() / np.abs(b).max()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    nrm = float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))
    return rel, cos, nrm


print("\n=== NPU output vs fp64 models ===")
rn = np.array([stats(got[i], narrowed) for i in range(M)])
re_ = np.array([stats(got[i], exact) for i in range(M)])
print(f"  vs bf16-narrowed model (encoder check):")
print(f"    maxrel worst {rn[:,0].max():.3e}   cos worst {rn[:,1].min():.9f}   nrmse worst {rn[:,2].max():.3e}")
print(f"  vs exact GGUF model (total transform cost):")
print(f"    maxrel worst {re_[:,0].max():.3e}   cos worst {re_[:,1].min():.9f}   nrmse worst {re_[:,2].max():.3e}")
enc_ok = rn[:, 0].max() < 5e-3
print(f"\n  encoder is kernel-compatible: {'YES' if enc_ok else 'NO'} "
      f"(bar: maxrel < 5e-3 vs the narrowed model, matching check.py's q4k bar)")
print(f"  all {M} tokens identical: {bool(np.allclose(got, got[0]))}")
