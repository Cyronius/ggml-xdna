"""Test vectors for gemm_q4_prefill at an arbitrary (N, K, T).

openflowlm-next's own designs/gemm_q4_prefill/make_test.py only knows three
shapes from its task table, none of which is this model's (K=2048, N=8192). This
generates the same buffers for any shape by importing that repo's packers
read-only, so nothing there is edited. Outputs land in the design directory,
which its .gitignore already covers (designs/*/*.bin, designs/*/*.cfg).

  python gqp_vectors.py --n 8192 --k 2048 --t 256 --build build_n8192_k2048_t256

Reference: the pool is dequantised ONCE into a dense [N,K] matrix and multiplied
by all T activations at once. make_test.py calls pool_reference per token, which
is 2048 chunk-dequants x T -- far too slow at T=256. The dense path is checked
against pool_reference on one token.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

OK = Path(r"C:\code\openflowlm-next\open_kernels")
DESIGN = OK / "designs" / "gemm_q4_prefill"
sys.path.insert(0, str(OK))
sys.path.insert(0, str(OK.parent / "npu_offload" / "gemm_rtp"))
from q4_1_pack import (CH, chunk_geometry, dequant_chunk,  # noqa: E402
                       pack_q4_1_pool, pool_reference, random_q4_1_blocks)
from npue import tile_b  # noqa: E402

K_TILE, MAC_S, MAC_T = 64, 8, 8


def dense_from_pool(pool: np.ndarray, n: int, k: int, rs: int) -> np.ndarray:
    """Dequantise the whole pool once into [n, k] float32."""
    nch, _, rows0, cols0 = chunk_geometry(n, k, rs)
    w = np.zeros((n, k), np.float32)
    for c in range(nch):
        r0, c0 = int(rows0[c]), int(cols0[c])
        w[r0:r0 + 32, c0:c0 + 256] = dequant_chunk(pool[c * CH:(c + 1) * CH])
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--t", type=int, default=256)
    ap.add_argument("--tile-n", type=int, default=32)
    ap.add_argument("--build", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    n, k, t = a.n, a.k, a.t
    if t % (a.tile_n * 8):
        sys.exit(f"T={t} must be a multiple of tile_n*8={a.tile_n * 8}")
    if n % 256 or k % 256:
        sys.exit(f"({n},{k}) must tile in 256-row blocks / 256-K bands")
    tag = a.tag or f"n{n}_k{k}_t{t}"
    rng = np.random.default_rng(a.seed)

    print(f"packing q4_1 pool [{n},{k}] ...")
    pool = pack_q4_1_pool(random_q4_1_blocks(n, k, rng), rs=2)

    print(f"drawing {t} distinct bf16 activations, K={k} ...")
    xs = [rng.standard_normal(k).astype(np.float32).astype(bfloat16) for _ in range(t)]
    assert len({x.tobytes() for x in xs}) == t, "activations collided"
    x_tk = np.stack(xs)

    print("dequantising pool once -> dense [N,K] ...")
    w = dense_from_pool(pool, n, k, rs=2)
    chk = pool_reference(pool, xs[0].astype(np.float32), n, k, rs=2)
    mine = (w.astype(np.float64) @ xs[0].astype(np.float64)).astype(np.float32)
    d = float(np.abs(mine - chk).max() / (np.abs(chk).max() + 1e-30))
    print(f"  dense-vs-pool_reference on token 0: maxrel {d:.3e}")
    if d > 1e-6:
        sys.exit("dense dequant disagrees with pool_reference")

    print(f"reference [N,T] = W @ X^T ...")
    ref_dev = (w.astype(np.float64) @ x_tk.astype(np.float64).T).astype(np.float32)

    x_kt = np.ascontiguousarray(x_tk.T)
    x_tiled = tile_b(x_kt.view(np.uint16), K_TILE, a.tile_n, MAC_S, MAC_T, order="k,n").view(bfloat16)

    (DESIGN / f"w_{tag}.bin").write_bytes(pool.tobytes())
    (DESIGN / f"x_{tag}.bin").write_bytes(x_tiled.tobytes())
    (DESIGN / f"ref_{tag}.bin").write_bytes(ref_dev.tobytes())

    cfg = ["device",
           f"xclbin G {a.build}/final.xclbin",
           f"kernelx k G {a.build}/insts.bin",
           f"buf w {pool.nbytes} w_{tag}.bin",
           f"buf x {x_tiled.nbytes} x_{tag}.bin",
           f"buf y {ref_dev.nbytes}"]
    cfg += ["run k w x y"] * a.runs
    cfg += [f"dump y y_{tag}.bin {ref_dev.nbytes}", ""]
    (DESIGN / f"run_{tag}.cfg").write_text("\n".join(cfg), newline="\n")

    print(f"\n{tag}: N={n} K={k} T={t}")
    print(f"  w {pool.nbytes} B   x {x_tiled.nbytes} B   y {ref_dev.nbytes} B")
    print(f"  wrote run_{tag}.cfg -> {a.build}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
