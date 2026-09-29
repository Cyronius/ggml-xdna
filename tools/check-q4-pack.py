"""Cross-check our C++ weight packer against openflowlm-next's own.

`test-q4-pack --dump DIR` writes each tensor's raw GGUF bytes and the pool our
C++ produced. This runs openflowlm-next's `q4_1_pack.pack_q4_1_pool` on the
same raw bytes and compares byte for byte.

openflowlm-next is imported read-only and nothing there is written.

The reference packer takes Q4_1 blocks. Q4_0 has no minimum and subtracts 8
from every code instead, so this builds the equivalent Q4_1 blocks first: same
nibbles, same scale, minimum -8*d. That rewrite is done here independently of
the C++ so agreement means something; that it is the right rewrite at all is
what the C++ side checks against ggml's own dequantisation.

  python tools/check-q4-pack.py <dump-dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OK = Path(r"C:\code\openflowlm-next\open_kernels")
sys.path.insert(0, str(OK))
from q4_1_pack import pack_q4_1_pool  # noqa: E402


def q4_0_to_q4_1_blocks(raw: np.ndarray, n: int, k: int) -> np.ndarray:
    """uint8[n*(k/32)*18] -> Q4_1-shaped blocks uint8[n, k/32, 20]."""
    nb = k // 32
    b = raw.reshape(n, nb, 18)
    d = b[:, :, 0:2].copy().view(np.float16)          # [n, nb, 1]
    m = (-8.0 * d.astype(np.float32)).astype(np.float16)
    out = np.empty((n, nb, 20), np.uint8)
    out[:, :, 0:2] = d.view(np.uint8)
    out[:, :, 2:4] = m.view(np.uint8)
    out[:, :, 4:20] = b[:, :, 2:18]                   # nibbles are untouched
    return out


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    d = Path(sys.argv[1])

    metas = sorted(d.glob("*.txt"))
    if not metas:
        sys.exit(f"no dumps in {d}")

    failures = 0
    for meta_path in metas:
        meta = dict(line.split(maxsplit=1) for line in meta_path.read_text().strip().splitlines())
        name, ty = meta["name"], meta["type"]
        k, n, rs = int(meta["k"]), int(meta["n"]), int(meta["rs"])
        stem = meta_path.with_suffix("")

        raw = np.fromfile(stem.with_suffix(".raw"), np.uint8)
        ours = np.fromfile(stem.with_suffix(".pool"), np.uint8)

        if ty == "q4_1":
            blocks = raw.reshape(n, k // 32, 20)
        elif ty == "q4_0":
            blocks = q4_0_to_q4_1_blocks(raw, n, k)
        else:
            sys.exit(f"unhandled type {ty}")

        theirs = pack_q4_1_pool(blocks, rs)

        same_size = ours.size == theirs.size
        identical = same_size and bool(np.array_equal(ours, theirs))
        if not identical:
            failures += 1
            where = "size differs" if not same_size else \
                f"first of {int((ours != theirs).sum())} differing bytes at {int(np.argmax(ours != theirs))}"
            print(f"FAIL {ty:5s} {name}: {where}")
        else:
            print(f"PASS {ty:5s} {name}: {ours.size} bytes identical to the reference packer")

    print("FAILED" if failures else "ALL PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
