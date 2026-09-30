// Operands for the NPU's bfp16ebs8 matmul (kernels/bfp16_gemm): blocks of
// eight values sharing one exponent byte, each value a signed 8-bit mantissa,
// nine bytes a block, rounded to nearest on the host.
//
// The kernel computes C[M x N] = A[M x K] * B[K x N]. A is the activations,
// one token per row; B is given as its transpose, which is exactly a GGUF
// weight's layout (one output per row, K along the row). Blocks run along K
// in both. Each operand is laid out tile by tile in the order the cores read
// it, so every NPU transfer is one contiguous run:
//
//   A: for each block of m rows, for each chunk of k along K: one m x k tile
//   B: for each of the array's `cols` columns c, for each of that column's
//      n-row groups (rows (c + cols*t)*n onward), for each chunk of k: one
//      n x k tile
//
// Inside a tile the 8x8 subtiles go row pair by row pair (16 rows), block
// column by block column, the pair's first eight rows then its second. The
// kernel reads rows z and z+1 from one stream that way.
//
// Traces: HYBRID-BFP16-NUMERICS (the rounding), HYBRID-Q4-PACK (the weight path)
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

constexpr size_t BFP16_BLOCK_BYTES = 9;

// One block of eight: exponent byte = the largest of the eight float
// exponents, mantissa q = round-to-nearest-even(x * 2^(133 - exponent)),
// saturated to [-128, 127]. A value that rounds up to +128 saturates rather
// than raising the exponent, which costs only that value.
void bfp16_encode_block(const float * x, uint8_t * out);
void bfp16_decode_block(const uint8_t * in, float * x);

struct bfp16_tiling {
    int m = 128, k = 64, n = 64;  // tile sizes the design was built with
    int cols = 8;                 // array columns the design uses
    int rows = 4;                 // and rows
};

// Activations, M x K row-major float -> A in the kernel's order. M a multiple
// of t.m, K of t.k. `out` is resized to M*K/8*9.
void bfp16_pack_a(const float * x, int64_t M, int64_t K, const bfp16_tiling & t, std::vector<uint8_t> & out,
                  int n_threads);

// Weights, N x K row-major float (a GGUF tensor as dequantised) -> B in the
// kernel's order. N a multiple of t.n * t.cols, K of t.k.
void bfp16_pack_b(const float * w, int64_t N, int64_t K, const bfp16_tiling & t, std::vector<uint8_t> & out,
                  int n_threads);

// Runs fn(begin, end) over [0, n), in parallel however the caller likes
// (thread_pool.h).
using bfp16_runner = std::function<void(int64_t n, const std::function<void(int64_t, int64_t)> & fn)>;

// bfp16_pack_a straight into `out` (M*K/8*9 bytes, e.g. a mapped NPU buffer).
void bfp16_pack_a_into(const float * x, int64_t M, int64_t K, const bfp16_tiling & t, uint8_t * out,
                       const bfp16_runner & run);

// bfp16_pack_b straight into `out` (N*K/8*9 bytes).
void bfp16_pack_b_into(const float * w, int64_t N, int64_t K, const bfp16_tiling & t, uint8_t * out,
                       const bfp16_runner & run);

// Where the block holding columns [8*cb, 8*cb + 8) of row r goes: its byte
// offset in an A operand of K columns, or a B operand of N rows and K
// columns. For producers that encode blocks straight into place, in whatever
// order they compute them.
size_t bfp16_a_offset(int64_t r, int64_t cb, int64_t K, const bfp16_tiling & t);
size_t bfp16_b_offset(int64_t r, int64_t cb, int64_t N, int64_t K, const bfp16_tiling & t);

// The same offsets for a whole row, without a division per block: calls
// fn(cb, byte offset) for cb = 0 .. K/8-1 in order.
template <typename F>
inline void bfp16_row_blocks(int64_t first_tile, int64_t rr, int64_t K, int64_t tile_rows, const bfp16_tiling & t, F && fn) {
    const int64_t tkb = t.k / 8, nk = K / t.k, tile = tile_rows * tkb;
    int64_t at = first_tile * nk * tile + (rr / 16) * tkb * 16 + rr % 16;
    for (int64_t kk = 0, cb = 0; kk < nk; kk++, at += tile)
        for (int64_t x = 0; x < tkb; x++, cb++) fn(cb, (size_t) (at + x * 16) * BFP16_BLOCK_BYTES);
}
template <typename F>
inline void bfp16_a_row(int64_t r, int64_t K, const bfp16_tiling & t, F && fn) {
    bfp16_row_blocks(r / t.m, r % t.m, K, t.m, t, fn);
}
template <typename F>
inline void bfp16_b_row(int64_t r, int64_t N, int64_t K, const bfp16_tiling & t, F && fn) {
    const int64_t g = r / t.n, per_col = N / t.n / t.cols;
    bfp16_row_blocks((g % t.cols) * per_col + g / t.cols, r % t.n, K, t.n, t, fn);
}

// The output C of a --c-tiled build (kernels/bfp16_gemm/designs/
// whole_array_bfp_rtp.py) is not row-major. Each array column's results land
// as one contiguous run, which the NPU writes up to twice as fast: column c
// owns floats [c*M*N/cols, (c+1)*M*N/cols); in it come blocks of m*rows x n,
// row block by row block, then the column's N tiles c, c+cols, ... in order,
// each block row-major. This copies the first `n_cols` columns (a multiple of
// t.n) of row r into `row`.
void bfp16_c_row(const float * c, int64_t M, int64_t N, const bfp16_tiling & t, int64_t r, int64_t n_cols, float * row);

// The output of a build with a 16-bit output mode (whole_array_bfp_rtp.py
// --out-mode, --c-tiled): each core packs its m x n tile's bf16 results at
// the tile's front, 8x8 block after 8x8 block in the tile's own block order,
// and the tile then travels as fp32, so the bf16 pairs land wherever those
// fp32 words do. Mode 1 is every value; mode 2 is SiLU(gate) * up, n/2
// values per tile, from weights ordered by bfp16_interleave_gate_up. This
// copies row r's results (N of them in mode 1, N/2 in mode 2) into `row` as
// floats.
void bfp16_c16_row(const float * c, int64_t M, int64_t N, const bfp16_tiling & t, int mode, int64_t r, float * row);

// gate and up (F x K each, row-major) as one 2F x K weight for output mode
// 2: 8 gate rows, then the 8 up rows that pair with them, and so on.
void bfp16_interleave_gate_up(const float * gate, const float * up, int64_t F, int64_t K, float * out);

// Inverse of bfp16_pack_a / bfp16_pack_b, back to row-major floats (tests).
void bfp16_unpack_a(const uint8_t * in, int64_t M, int64_t K, const bfp16_tiling & t, float * x);
void bfp16_unpack_b(const uint8_t * in, int64_t N, int64_t K, const bfp16_tiling & t, float * w);
