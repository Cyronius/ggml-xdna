# whole_array.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Whole-array bfp16 x bfp16 -> fp32 GEMM whose core program serves every shape.

npu-prefill-engine variant of whole_array_bfp_acc.py (--acc 33: fp32 sums kept
in the single-buffered fp32 output tile). The core program no longer bakes in
the shape:
  * K: the runtime sequence writes K/k - 1 into a per-core runtime parameter
    (Buffer(use_write_rtp=True)); the core reads it once per output tile, right
    after acquiring that tile's first A/B chunk. That chunk can only arrive
    after this dispatch's parameter write, which precedes its DMAs in the
    instruction stream.
  * M, N: the core body is one output tile; IRON's implicit while(true) repeats
    it for however many tiles the runtime sequence feeds.
  * the output step: the runtime sequence also writes an output mode, and after
    the last K chunk the core runs out_epilogue on the finished tile: 0 leaves
    it fp32, 1 packs it as bf16, 2 packs SiLU(gate) * up as bf16 (--out-mode).
So one xclbin serves any (M, K, N) with M % 4m == 0, N % 8n == 0, K % k == 0,
in any output mode; only insts.bin changes. Kernels: aie_kernels/aie2p/
mm_bfp_acc.cc (F32_FIRST/MID, OUT_EPILOGUE).
"""

import argparse
import os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import (
    Buffer,
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    StreamDims,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import (
    add_compile_args,
    device_from_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

_KERNEL_SRC = (
    Path(__file__).resolve().parents[1] / "aie_kernels" / "aie2p" / "mm_bfp_acc.cc"
)


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def whole_array_matmul(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int] = 512,
    K: CompileTime[int] = 512,
    N: CompileTime[int] = 512,
    m: CompileTime[int] = 64,
    k: CompileTime[int] = 64,
    n: CompileTime[int] = 64,
    n_aie_cols: CompileTime[int] = 4,
    c_tiled: CompileTime[bool] = False,
    b_groups: CompileTime[int] = 0,
    b_kfull: CompileTime[int] = 0,
    out_mode: CompileTime[int] = 0,
):
    acc = 33  # fp32 sums in the output tile, fp32 out
    c_depth = 1
    c_dt = np.float32
    assert K // k >= 2
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols
    fifo_depth = 2

    assert M % (m * n_aie_rows) == 0, "M must be tileable into (m*n_aie_rows, k) blocks"
    assert K % k == 0
    assert N % (n * n_aie_cols) == 0, "N must be tileable into (k, n*n_aie_cols) blocks"

    # bfp16ebs8 matmul mac unit is 8x8x8; m/k/n must be multiples of these.
    r = s = t = 8
    assert m % r == 0, f"m ({m}) must be a multiple of {r}"
    assert k % s == 0, f"k ({k}) must be a multiple of {s}"
    assert n % t == 0, f"n ({n}) must be a multiple of {t}"

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    n_shim_mem_A = n_aie_rows if n_aie_cols > n_aie_rows else n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_l2_ty = np.ndarray[(m * k // 8 * n_A_tiles_per_shim,), np.dtype[v8bfp16ebs8]]
    B_l2_ty = np.ndarray[(k * n // 8,), np.dtype[v8bfp16ebs8]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[c_dt]]
    A_l1_ty = np.ndarray[(m, k // 8), np.dtype[v8bfp16ebs8]]
    B_l1_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[c_dt]]
    Acc_ty = np.ndarray[(m * n,), np.dtype[np.float32]]

    kernel_flags = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}"]
    # Local change: extra kernel defines, e.g. BFP_KFLAGS="-DROUND_CONV_EVEN".
    kernel_flags += os.environ.get("BFP_KFLAGS", "").split()

    def _kern(name, macro, arg_types):
        return ExternalFunction(
            name,
            source_file=str(_KERNEL_SRC),
            arg_types=arg_types,
            compile_flags=kernel_flags + [f"-D{macro}"],
        )

    k_first = _kern("mm_f32_first", "F32_FIRST", [A_l1_ty, B_l1_ty, C_l1_ty])
    k_mid = _kern("mm_f32_mid", "F32_MID", [A_l1_ty, B_l1_ty, C_l1_ty])
    k_out = _kern("out_epilogue", "OUT_EPILOGUE", [C_l1_ty, np.int32])
    assert out_mode in (0, 1, 2)

    A_l3l2_fifos: list[ObjectFifo] = []
    A_l2l1_fifos: list[ObjectFifo] = []
    B_l3l2_fifos: list[ObjectFifo] = []
    B_l2l1_fifos: list[ObjectFifo] = []
    C_l1l2_fifos: list[list[ObjectFifo]] = [[] for _ in range(n_aie_rows)]
    C_l2l3_fifos: list[ObjectFifo] = []

    for i in range(n_shim_mem_A):
        a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k // 8 * j for j in range(stop_row - start_row)]
        a_tmp_fifos = a_l3l2.cons().split(
            of_offsets,
            obj_types=[A_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
        )
        A_l2l1_fifos.extend(a_tmp_fifos)

    for col in range(n_aie_cols):
        b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}")
        )

        # 8x8 blocks -> row-major on the way to DDR (from whole_array_mixed)
        c_l2l3_dims: StreamDims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        c_l2l3 = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, dims_to_stream=c_l2l3_dims
        )
        C_l2l3_fifos.append(c_l2l3)
        of_offsets = [m * n * i for i in range(n_aie_rows)]
        c_tmp_fifos = c_l2l3.prod().join(
            of_offsets,
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[c_depth] * n_aie_rows,
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j].append(c_tmp_fifos[j])

    def _chunk(in_a, in_b, kern, *rest):
        elem_in_a = in_a.acquire(1)
        elem_in_b = in_b.acquire(1)
        kern(elem_in_a, elem_in_b, *rest)
        in_a.release(1)
        in_b.release(1)

    def core_fn(in_a, in_b, out_c, rtp, first, mid, epilogue):
        # One output tile per pass of the implicit while(true).
        elem_out = out_c.acquire(1)
        elem_in_a = in_a.acquire(1)
        elem_in_b = in_b.acquire(1)
        # this dispatch's K/k - 1 and output mode
        n_mid = rtp[0]
        mode = rtp[1]
        first(elem_in_a, elem_in_b, elem_out)
        in_a.release(1)
        in_b.release(1)
        for _ in range_(n_mid):
            _chunk(in_a, in_b, mid, elem_out)
        epilogue(elem_out, mode)
        out_c.release(1)

    rtps = [
        [
            Buffer(
                np.ndarray[(2,), np.dtype[np.int32]],
                name=f"nmid_{row}_{col}",
                initial_value=np.array([K // k - 1, out_mode], dtype=np.int32),
                use_write_rtp=True,
            )
            for col in range(n_aie_cols)
        ]
        for row in range(n_aie_rows)
    ]

    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                rtps[row][col],
                k_first,
                k_mid,
                k_out,
            ],
            stack_size=0x1000,  # out_epilogue's SiLU frame is ~3.2 KB
        ),
    )

    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    # --b-groups NF / --b-kfull KF: B is a buffer made for more rows (NF) or
    # longer rows (KF) than this call reads, laid out so that the rows or
    # columns it reads come first. See B_tiles below.
    b_rows, b_cols = (b_groups or N), (b_kfull or K)
    B_ty = np.ndarray[(b_rows * b_cols // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N,), np.dtype[c_dt]]

    tb_max_n_rows = 4
    # Local change: a C drain covering two row blocks strides m*4*N bf16 elements,
    # which overflows the shim BD's 2^20-word stride at N=12288. Drain one row
    # block per task group there.
    if m * n_aie_rows * N * np.dtype(c_dt).itemsize // 4 > (1 << 20):
        tb_max_n_rows = 2
    # Local change: was tb_max_n_rows // 2, which needs M >= 8*m (M=256 failed at m=64).
    tb_n_rows = min(tb_max_n_rows // 2, M // m // n_aie_rows)

    # Local change (whole_array_bfp_lin): A and B arrive pre-tiled from the host,
    # each tile contiguous and the tiles in the order the cores consume them:
    #   A: row block rb (m rows), then K block kk        -> offset rb * m * K/8
    #   B: column c, then N block t (rows (c+8t)*n..), then K block kk
    # so every shim transfer is one long contiguous read instead of one
    # k*9/8-byte row per DMA step at a K*9/8-byte stride.
    def _lin_tap(total, offset, length, repeat=1):
        d0 = 256  # elements (2304 B); every length here is a multiple of it
        rows = length // d0
        assert length % d0 == 0
        d1 = max(d for d in range(1, 1024) if rows % d == 0)
        d2 = rows // d1
        assert d2 < 1024 and repeat <= 64
        return TensorAccessPattern(
            (total,), offset, [repeat, d2, d1, d0], [0, d1 * d0, d0, 1]
        )

    A_tiles = [
        _lin_tap(M * K // 8, rb * m * (K // 8), m * (K // 8), N // n // n_aie_cols)
        for rb in range(M // m)
    ]
    B_tiles = [
        _lin_tap(N * K // 8, c * (N // n_aie_cols) * (K // 8), (N // n_aie_cols) * (K // 8))
        for c in range(n_aie_cols)
    ]
    # Each column still consumes its n-row groups t = 0, 1, ... and in each
    # all nk K-chunk tiles; only where they sit in the buffer changes.
    #   --b-groups NF: groups of n*cols rows, one after another, in each the
    #     columns' tiles in turn; so the first N rows of an NF-row buffer are
    #     one prefix, and one buffer serves every N (attention's keys).
    #   --b-kfull KF: the usual layout at K = KF; this call reads the first nk
    #     tiles of each group's KF/k (attention's values, transposed).
    tb = n * k // 8  # blocks per tile
    nk, per_col = K // k, N // n // n_aie_cols
    if b_groups or b_kfull:
        assert not (b_groups and b_kfull) and (nk * tb) % 256 == 0
        d1 = nk * tb // 256
        assert d1 < 1024 and per_col < 1024
        if b_groups:
            assert N <= b_groups
            step, first = n_aie_cols * nk * tb, lambda c: c * nk * tb
        else:
            assert K <= b_kfull
            step = (b_kfull // k) * tb
            first = lambda c: c * per_col * step
        B_tiles = [
            TensorAccessPattern(
                (b_rows * b_cols // 8,), first(c), [1, per_col, d1, 256], [0, step, 256, 1]
            )
            for c in range(n_aie_cols)
        ]
    C_tiles = TensorTiler2D.step_tiler(
        (M, N),
        (m * n_aie_rows, n),
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
    )
    c_index = 0

    # --c-tiled: C lands in DDR as each column's memtile streams it, one
    # contiguous run per drain, instead of scattered into row-major C (one
    # n-element row per DMA step at an N-element stride). Column c owns
    # [c*M*N/cols, (c+1)*M*N/cols); inside it the (m*rows x n) blocks go row
    # block by row block, then that column's N tiles (c, c+cols, ...) in
    # order, each block row-major. Changes only the runtime sequence, so the
    # insts.bin runs under the row-major build's xclbin.
    c_blk = m * n_aie_rows * n
    c_per_rb = (N // n // n_aie_cols) * c_blk

    def sequence(a, b, c, A_prods, B_prods, C_conses):
        nonlocal c_index
        # This dispatch's K-chunk count and output mode, written before any of
        # its DMAs start.
        for row in rtps:
            for rtp in row:
                rtp[0] = K // k - 1
                rtp[1] = out_mode
        tg = TaskGroup()
        for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                )
                for col in range(n_aie_cols):
                    c_tap = C_tiles[c_index]
                    if c_tiled:
                        c_tap = _lin_tap(
                            M * N,
                            col * (M * N // n_aie_cols) + row_base * c_per_rb,
                            current_tb_n_rows * c_per_rb,
                        )
                    C_conses[col].drain(
                        c,
                        tap=c_tap,
                        wait=True,
                        group=tg,
                    )
                    c_index += 1
                    for tile_row in range(current_tb_n_rows):
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)
                        if col < n_aie_rows:
                            A_prods[col].fill(
                                a,
                                tap=A_tiles[tile_offset],
                                group=tg,
                            )
                        B_prods[col].fill(
                            b,
                            tap=B_tiles[col],
                            group=tg,
                        )
                if tb > 0 or (tb == 0 and pingpong > 0):
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    rt = Runtime(
        sequence,
        [
            A_ty,
            B_ty,
            C_ty,
            [f.prod() for f in A_l3l2_fifos],
            [f.prod() for f in B_l3l2_fifos],
            [f.cons() for f in C_l2l3_fifos],
        ],
    )

    return Program(
        iron.get_current_device(),
        rt,
        workers=[w for row in workers for w in row],
    ).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="AIE Whole-Array bfp16ebs8 Matmul, fp32, runtime K",
    )
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=64)
    p.add_argument(
        "--n-aie-cols", dest="n_aie_cols", type=int, choices=[1, 2, 4, 8], default=4
    )
    p.add_argument("--c-tiled", dest="c_tiled", action="store_true")
    p.add_argument("--b-groups", dest="b_groups", type=int, default=0)
    p.add_argument("--b-kfull", dest="b_kfull", type=int, default=0)
    p.add_argument("--out-mode", dest="out_mode", type=int, choices=[0, 1, 2], default=0)

    return p


def _compile_kwargs(opts):
    return dict(
        M=opts.M,
        K=opts.K,
        N=opts.N,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        c_tiled=opts.c_tiled,
        b_groups=opts.b_groups,
        b_kfull=opts.b_kfull,
        out_mode=opts.out_mode,
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        whole_array_matmul,
        opts,
        compile_kwargs=_compile_kwargs,
        device=lambda o: device_from_args(o, n_cols=o.n_aie_cols),
    )


if __name__ == "__main__":
    main()
