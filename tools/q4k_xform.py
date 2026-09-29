"""Stage 0a: can GGUF Q4_K bytes feed the gemm_q4 GEMM_Q4K kernel, and at what cost?

Part 1 validates that this file understands the 4736 B chunk layout, by decoding
the kernel's own w_q4k_pool.bin and reproducing make_q4k.py's ref_q4k.bin.
Part 2 reads a real Q4_K tensor from the GGUF, encodes it into that layout, and
measures what the transform costs against an fp64 reference taken from the GGUF
bytes themselves.

No NPU involved. Run: python q4k_xform.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

GEMM = Path(r"C:\code\openflowlm-next\open_kernels\designs\gemm_q4")
GEMV = Path(r"C:\code\openflowlm-next\open_kernels\designs\gemv_q4")
GGUF = Path(r"C:\Users\josha\.flm\models\Qwen3.6-27B-A2.8B-open\qwen36-27b-a2.8b-mtp-Q4KM.gguf")
sys.path.insert(0, r"C:\code\npu-prefill-engine\third_party\src-tmp\llama.cpp-b10944\gguf-py")

N, K = 8192, 2048
BAND_ROWS, RS, ROWS, KB, KT = 64, 2, 32, 8, 256
BANDS = N // BAND_ROWS                  # 128
PER_BAND = RS * K // 256                # 16 chunks per band
Q4K_BYTES = 4736
PERM = np.concatenate([np.arange(0, ROWS, 2), np.arange(1, ROWS, 2)])
INV_PERM = np.argsort(PERM)

# chunk field offsets
O_D, O_M, O_SC, O_MN, O_NIB = 0, 64, 128, 384, 640


def nib_index(r: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Flat nibble index inside a chunk's 4096 B nibble block for row r, column k.

    make_q4k.py builds it as stack([lo, hi]) -> (half, k, r16), so nibble n sits
    in byte n//2, low half when n is even.
    """
    return (r // 16) * 4096 + k * 16 + (r % 16)


def decode_chunks(buf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[BANDS][PER_BAND][4736] -> (values[N, K] as float64, row index per band).

    Inverse of make_q4k.py's writer, so agreement with ref_q4k.bin proves the
    layout reading here is right.
    """
    c = buf.reshape(BANDS, PER_BAND, Q4K_BYTES)
    out = np.zeros((N, K), np.float64)

    dsup = c[:, :, O_D:O_D + 64].copy().view(bfloat16).astype(np.float64)      # [b][c][32] PERM order
    mco = c[:, :, O_M:O_M + 64].copy().view(bfloat16).astype(np.float64)
    sc = c[:, :, O_SC:O_SC + 256].reshape(BANDS, PER_BAND, KB, ROWS)           # [b][c][kb][32] PERM
    mn = c[:, :, O_MN:O_MN + 256].reshape(BANDS, PER_BAND, KB, ROWS)
    nb = c[:, :, O_NIB:]                                                       # [b][c][4096]

    # undo the lane permutation
    dsup = dsup[:, :, INV_PERM]
    mco = mco[:, :, INV_PERM]
    sc = sc[:, :, :, INV_PERM].astype(np.float64)
    mn = mn[:, :, :, INV_PERM].astype(np.float64)

    rr, kk = np.meshgrid(np.arange(ROWS), np.arange(KT), indexing="ij")
    n = nib_index(rr, kk)
    byte, hi = n // 2, (n % 2).astype(bool)

    for b in range(BANDS):
        for ci in range(PER_BAND):
            rs, kt = ci % RS, ci // RS
            raw = nb[b, ci][byte]
            q = np.where(hi, raw >> 4, raw & 0x0F).astype(np.float64)          # [32][256]
            s = sc[b, ci].repeat(32, axis=0).reshape(KB * 32, ROWS)[:, :]      # placeholder
            s = np.repeat(sc[b, ci], 32, axis=0).T                            # [32][256]
            m = np.repeat(mn[b, ci], 32, axis=0).T
            r0 = b * BAND_ROWS + rs * ROWS
            out[r0:r0 + ROWS, kt * KT:(kt + 1) * KT] = (
                dsup[b, ci][:, None] * s * q + mco[b, ci][:, None] * m)
    return out


def encode_chunks(d: np.ndarray, dmin: np.ndarray, sc: np.ndarray,
                  mn: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row Q4_K fields -> [BANDS][PER_BAND][4736] chunk pool.

    d, dmin:  [N][K//256] float   (GGUF superblock scales; dmin positive)
    sc, mn:   [N][K//256][8] uint8 (6-bit sub-values, unpacked to bytes)
    q:        [N][K] uint8         (nibble values 0..15)
    """
    out = np.zeros((BANDS, PER_BAND, Q4K_BYTES), np.uint8)
    rr, kk = np.meshgrid(np.arange(ROWS), np.arange(KT), indexing="ij")
    n = nib_index(rr, kk)
    byte, hi = n // 2, (n % 2).astype(bool)

    for b in range(BANDS):
        for ci in range(PER_BAND):
            rs, kt = ci % RS, ci // RS
            r0 = b * BAND_ROWS + rs * ROWS
            rows = slice(r0, r0 + ROWS)

            # kernel stores +mco*mn where GGUF means -dmin*m, so mco is negated
            out[b, ci, O_D:O_D + 64] = (
                d[rows, kt].astype(bfloat16)[PERM].view(np.uint8))
            out[b, ci, O_M:O_M + 64] = (
                (-dmin[rows, kt]).astype(bfloat16)[PERM].view(np.uint8))
            out[b, ci, O_SC:O_SC + 256] = sc[rows, kt][PERM].T.reshape(-1)
            out[b, ci, O_MN:O_MN + 256] = mn[rows, kt][PERM].T.reshape(-1)

            blk = q[rows, kt * KT:(kt + 1) * KT]                               # [32][256]
            packed = np.zeros(4096, np.uint8)
            np.add.at(packed, byte[~hi], blk[~hi])
            np.add.at(packed, byte[hi], (blk[hi] << 4).astype(np.uint8))
            out[b, ci, O_NIB:] = packed
    return out


def unpack_q4k(raw: np.ndarray, nrow: int, ncol: int):
    """GGUF block_q4_K bytes -> (d, dmin, sc, mn, q) with q in ggml's weight order."""
    nsb = ncol // 256
    b = raw.reshape(nrow, nsb, 144)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float64)[:, :, 0]
    dmin = b[:, :, 2:4].copy().view(np.float16).astype(np.float64)[:, :, 0]
    s12 = b[:, :, 4:16]
    qs = b[:, :, 16:]

    # get_scale_min_k4: j<4 -> (s[j]&63, s[j+4]&63);
    # else (s[j+4]&0xF | (s[j-4]>>6)<<4, s[j+4]>>4 | (s[j]>>6)<<4)
    sc = np.zeros((nrow, nsb, 8), np.uint8)
    mn = np.zeros((nrow, nsb, 8), np.uint8)
    for j in range(4):
        sc[:, :, j] = s12[:, :, j] & 63
        mn[:, :, j] = s12[:, :, j + 4] & 63
    for j in range(4, 8):
        sc[:, :, j] = (s12[:, :, j + 4] & 0x0F) | ((s12[:, :, j - 4] >> 6) << 4)
        mn[:, :, j] = (s12[:, :, j + 4] >> 4) | ((s12[:, :, j] >> 6) << 4)

    # ggml order: per 64-group, 32 low nibbles then 32 high nibbles of 32 bytes
    q = np.zeros((nrow, nsb, 256), np.uint8)
    for g in range(4):
        blk = qs[:, :, g * 32:(g + 1) * 32]
        q[:, :, g * 64:g * 64 + 32] = blk & 0x0F
        q[:, :, g * 64 + 32:(g + 1) * 64] = blk >> 4
    return d, dmin, sc, mn, q.reshape(nrow, ncol)


def ref_from_fields(d, dmin, sc, mn, q, x, cast=None):
    """fp64 matmul of the dequantised weights against x[K]."""
    nrow, ncol = q.shape
    nsb = ncol // 256
    dd = d if cast is None else d.astype(cast).astype(np.float64)
    mm = dmin if cast is None else (-(-dmin).astype(cast).astype(np.float64))
    scale = (dd[:, :, None] * sc.astype(np.float64)).repeat(32, axis=2).reshape(nrow, ncol)
    minv = (mm[:, :, None] * mn.astype(np.float64)).repeat(32, axis=2).reshape(nrow, ncol)
    w = scale * q.astype(np.float64) - minv
    return w @ x


def rel(a, b):
    den = np.maximum(np.abs(b).max(), 1e-30)
    return float(np.abs(a - b).max() / den)


print("=== Part 1: does this file understand the 4736 B layout? ===")
pool = np.fromfile(GEMM / "w_q4k_pool.bin", np.uint8)
xq = np.fromfile(GEMV / "x_qkv.bin", np.uint8).view(bfloat16).astype(np.float64)
ref = np.fromfile(GEMM / "ref_q4k.bin", np.float32).astype(np.float64)
vals = decode_chunks(pool)
mine = vals @ xq
print(f"  pool {pool.size} B, x {xq.size}, ref {ref.size}")
print(f"  decoded-vs-ref_q4k.bin  maxrel = {rel(mine, ref):.3e}")
ok = rel(mine, ref) < 1e-6
print(f"  layout understood: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit("layout mismatch - stop, everything below is meaningless")

print("\n=== Part 2: a real GGUF Q4_K tensor through the same layout ===")
import gguf
rd = gguf.GGUFReader(str(GGUF), "r")
# 16 attn_qkv + 11 attn_q are Q4_K at exactly (K=2048, N=8192), the built shape
t = next(t for t in rd.tensors
         if t.tensor_type.name == "Q4_K" and tuple(int(s) for s in t.shape) == (K, N))
print(f"  {t.name}  type={t.tensor_type.name}  shape={list(t.shape)}")
raw = np.asarray(t.data).reshape(N, -1)
assert raw.shape[1] == (K // 256) * 144, (raw.shape, (K // 256) * 144)
print(f"  {N} rows x {K} cols, {raw.shape[1]} B/row, no tiling or padding needed")

d, dmin, sc, mn, q = unpack_q4k(raw.reshape(-1), N, K)
print(f"  unpacked: d{d.shape} sc{sc.shape} q{q.shape}  "
      f"sc range {sc.min()}..{sc.max()}  mn range {mn.min()}..{mn.max()}")

rng = np.random.default_rng(0)
x = rng.standard_normal(K)
x_bf = np.asarray(x, dtype=bfloat16).astype(np.float64)   # kernel sees bf16 activations

exact = ref_from_fields(d, dmin, sc, mn, q, x_bf)                    # GGUF fp16 scales
narrowed = ref_from_fields(d, dmin, sc, mn, q, x_bf, cast=bfloat16)  # what a chunk can hold
print(f"  fp16->bf16 scale narrowing costs maxrel = {rel(narrowed, exact):.3e}")

chunks = encode_chunks(d, dmin, sc, mn, q)
round_trip = decode_chunks(chunks.reshape(-1)) @ x_bf
print(f"  encode->decode round trip vs narrowed  maxrel = {rel(round_trip, narrowed):.3e}")
print(f"  encode->decode round trip vs exact     maxrel = {rel(round_trip, exact):.3e}")

chunks.reshape(-1).tofile(GEMM / "w_q4k_gguf_pool.bin")
np.asarray(x, dtype=bfloat16).tofile(GEMM / "x_gguf.bin")
exact.astype(np.float32).tofile(GEMM / "ref_q4k_gguf.bin")
print(f"\n  wrote w_q4k_gguf_pool.bin ({chunks.size} B), x_gguf.bin, ref_q4k_gguf.bin")
print(f"  existing pool is {pool.size} B - match: {chunks.size == pool.size}")
