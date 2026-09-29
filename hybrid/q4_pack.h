// GGUF 4-bit weights into the chunk layout the open NPU kernels stream.
//
// The kernels read a weight matrix as 5120-byte chunks, each covering 32 rows
// and 256 columns of K, in a fixed order. Inside a chunk the scales come
// first, then the minimums, then the nibbles with two *rows* packed per byte
// rather than two columns, because the integer matrix unit wants its operand
// that way round.
//
// Nothing is requantised. GGUF's 4-bit nibbles and its per-32 scale and
// minimum carry through unchanged; the transform is a byte permutation plus
// narrowing the scale and minimum from 16-bit float to bfloat16. Q4_0 becomes
// the same shape exactly, because its value rule (q - 8) * d is (q * d) + m
// with m = -8 * d.
//
// This is a C++ port of openflowlm-next's `open_kernels/q4_1_pack.py`, and
// `hybrid/test-q4-pack.cpp` checks it produces byte-identical output to that
// file for real tensors. When the packer there changes, this has to follow.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

// Bytes per chunk, and the tile it covers.
constexpr size_t  Q4_CHUNK_BYTES = 5120;
constexpr int64_t Q4_CHUNK_ROWS  = 32;
constexpr int64_t Q4_CHUNK_COLS  = 256;
constexpr int64_t Q4_BLOCK       = 32;   // values sharing one scale, along K

// Where every chunk of an n-by-k matrix sits. `rs` is how many 32-row groups
// a band is split into: 2 for the standard layout, 4 for expert stripes.
struct q4_pool_geometry {
    int64_t n = 0, k = 0;
    int     rs = 2;
    int64_t per_band = 0;
    int64_t n_chunks = 0;

    size_t bytes() const { return (size_t) n_chunks * Q4_CHUNK_BYTES; }

    void chunk_origin(int64_t c, int64_t & row0, int64_t & col0) const {
        const int64_t band = c / per_band;
        const int64_t ci   = c % per_band;
        row0 = Q4_CHUNK_ROWS * rs * band + Q4_CHUNK_ROWS * (ci % rs);
        col0 = Q4_CHUNK_COLS * (ci / rs);
    }
};

// Fails when the shape does not tile: k must be a multiple of 256 and n a
// multiple of 32*rs.
bool q4_pool_geometry_init(int64_t n, int64_t k, int rs, q4_pool_geometry & g, std::string & err);

// `raw` is a GGUF tensor's bytes: n rows of k values, type Q4_0 or Q4_1
// (pass ggml's type id). `out` must be g.bytes() long.
bool q4_pack_pool(const void * raw, int ggml_type, const q4_pool_geometry & g, uint8_t * out, std::string & err);

// The pool read back as f32[n][k], row-major. For checking, not for speed.
void q4_dequant_pool(const uint8_t * pool, const q4_pool_geometry & g, float * out);

// 32-bit float to bfloat16, round to nearest with ties to even. Same rule as
// ggml's own conversion and as numpy's, which is what lets the checks compare
// against both. Exposed because they need to narrow a reference the same way.
inline uint16_t q4_f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t) ((u >> 16) | 0x0040);   // keep NaN a NaN
    }
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

inline float q4_bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
