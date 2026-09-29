#include "q4_pack.h"

#include "ggml.h"

namespace {

// A GGUF 4-bit block, read the same way for both types: 32 values along K,
// each value = code * d + m. Q4_0 has no stored minimum and subtracts 8 from
// every code instead, which is the same thing with m = -8 * d.
struct block_view {
    size_t bytes;       // per block
    size_t qs_offset;   // where the 16 nibble bytes start
    bool   has_min;
};

bool block_view_for(int type, block_view & bv, std::string & err) {
    switch (type) {
        case GGML_TYPE_Q4_0: bv = { 18, 2, false }; return true;
        case GGML_TYPE_Q4_1: bv = { 20, 4, true  }; return true;
        default:
            err = "only Q4_0 and Q4_1 can be packed without requantising; got ggml type "
                  + std::to_string(type);
            return false;
    }
}

inline float fp16_at(const uint8_t * p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return ggml_fp16_to_fp32((ggml_fp16_t) h);
}

// The code at (row r, column k) of the block row starting at `row`.
inline uint8_t code_at(const uint8_t * row, const block_view & bv, int64_t k) {
    const uint8_t * blk = row + (size_t) (k / Q4_BLOCK) * bv.bytes;
    const int64_t   j   = k % Q4_BLOCK;              // value within the block
    const uint8_t   b   = blk[bv.qs_offset + (j & 15)];
    return j < 16 ? (uint8_t) (b & 0x0F) : (uint8_t) (b >> 4);
}

// Where the code for (row-in-chunk r, column-in-chunk k) goes: the nibble
// index is (r/16)*4096 + k*16 + (r%16), two nibbles to a byte, the even one
// low. So consecutive rows within a half share a byte.
inline void nibble_slot(int64_t r, int64_t k, size_t & byte, bool & high) {
    const int64_t p = (r / 16) * 4096 + k * 16 + (r % 16);
    byte = 1024 + (size_t) (p / 2);
    high = (p & 1) != 0;
}

} // namespace

bool q4_pool_geometry_init(int64_t n, int64_t k, int rs, q4_pool_geometry & g, std::string & err) {
    if (rs != 2 && rs != 4) {
        err = "band row split must be 2 or 4, got " + std::to_string(rs);
        return false;
    }
    if (k % Q4_CHUNK_COLS != 0) {
        err = "K (" + std::to_string(k) + ") must be a multiple of 256";
        return false;
    }
    if (n % (Q4_CHUNK_ROWS * rs) != 0) {
        err = "rows (" + std::to_string(n) + ") must be a multiple of " + std::to_string(Q4_CHUNK_ROWS * rs);
        return false;
    }
    g.n = n;
    g.k = k;
    g.rs = rs;
    g.per_band = (int64_t) rs * k / Q4_CHUNK_COLS;
    g.n_chunks = n * k / (Q4_CHUNK_ROWS * Q4_CHUNK_COLS);
    return true;
}

bool q4_pack_pool(const void * raw, int type, const q4_pool_geometry & g, uint8_t * out, std::string & err) {
    block_view bv;
    if (!block_view_for(type, bv, err)) {
        return false;
    }
    const uint8_t * base = (const uint8_t *) raw;
    const size_t row_bytes = (size_t) (g.k / Q4_BLOCK) * bv.bytes;

    for (int64_t c = 0; c < g.n_chunks; c++) {
        int64_t r0, c0;
        g.chunk_origin(c, r0, c0);
        uint8_t * dst = out + (size_t) c * Q4_CHUNK_BYTES;
        std::memset(dst, 0, Q4_CHUNK_BYTES);

        for (int64_t r = 0; r < Q4_CHUNK_ROWS; r++) {
            const uint8_t * row = base + (size_t) (r0 + r) * row_bytes;

            // Scales and minimums, indexed block-major: slot = kb*32 + r.
            for (int64_t kb = 0; kb < Q4_CHUNK_COLS / Q4_BLOCK; kb++) {
                const uint8_t * blk = row + (size_t) ((c0 / Q4_BLOCK) + kb) * bv.bytes;
                const float d = fp16_at(blk);
                const float m = bv.has_min ? fp16_at(blk + 2) : -8.0f * d;
                const size_t slot = (size_t) (kb * Q4_CHUNK_ROWS + r);
                const uint16_t db = q4_f32_to_bf16(d);
                const uint16_t mb = q4_f32_to_bf16(m);
                std::memcpy(dst +       slot * 2, &db, 2);
                std::memcpy(dst + 512 + slot * 2, &mb, 2);
            }

            // Codes, transposed so two rows share a byte.
            for (int64_t k = 0; k < Q4_CHUNK_COLS; k++) {
                const uint8_t code = code_at(row, bv, c0 + k);
                size_t byte; bool high;
                nibble_slot(r, k, byte, high);
                dst[byte] |= high ? (uint8_t) (code << 4) : code;
            }
        }
    }
    return true;
}

void q4_dequant_pool(const uint8_t * pool, const q4_pool_geometry & g, float * out) {
    for (int64_t c = 0; c < g.n_chunks; c++) {
        int64_t r0, c0;
        g.chunk_origin(c, r0, c0);
        const uint8_t * src = pool + (size_t) c * Q4_CHUNK_BYTES;

        for (int64_t r = 0; r < Q4_CHUNK_ROWS; r++) {
            for (int64_t k = 0; k < Q4_CHUNK_COLS; k++) {
                size_t byte; bool high;
                nibble_slot(r, k, byte, high);
                const uint8_t code = high ? (uint8_t) (src[byte] >> 4) : (uint8_t) (src[byte] & 0x0F);

                const size_t slot = (size_t) ((k / Q4_BLOCK) * Q4_CHUNK_ROWS + r);
                uint16_t db, mb;
                std::memcpy(&db, src +       slot * 2, 2);
                std::memcpy(&mb, src + 512 + slot * 2, 2);

                out[(r0 + r) * g.k + (c0 + k)] = (float) code * q4_bf16_to_f32(db) + q4_bf16_to_f32(mb);
            }
        }
    }
}
