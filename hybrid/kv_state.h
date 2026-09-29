// The per-sequence cache state that llama.cpp hands out through
// llama_state_seq_get_data, taken apart into fields and put back together.
//
// This is the seam the hybrid design crosses: prefill produces keys and values
// somewhere else, and a llama.cpp context running on the GPU has to accept
// them. The public API takes an opaque blob; this is the blob's layout for the
// pinned release (b10944), a plain attention cache, one stream, no per-cell
// extensions. If the pinned release changes, this file is what has to follow.
//
// Layout, in order:
//   u32 magic = 0xaf143cd8, i32 seq_id    (the context's own header)
//   u32 n_stream                          (must be 1 here)
//   u32 cell_count
//   per cell:  i32 pos, u32 n_seq_id, i32 seq_id[n_seq_id]
//   u32 v_trans, u32 n_layer
//   per layer: i32 k_type, u64 k_row_bytes, u8 k[cell_count * k_row_bytes]
//   if !v_trans, per layer: i32 v_type, u64 v_row_bytes, u8 v[cell_count * v_row_bytes]
//   else,        per layer: i32 v_type, u32 v_el_bytes, u32 n_embd,
//                           u8 v[n_embd][cell_count * v_el_bytes]
//
// Keys are stored after rotary position encoding is applied, so whoever
// produces them has to apply it the same way llama.cpp does for the model.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

constexpr uint32_t KV_STATE_MAGIC = 0xaf143cd8u;

struct kv_state {
    int32_t  seq_id     = 0;
    uint32_t cell_count = 0;
    std::vector<int32_t>              pos;      // one per cell
    std::vector<std::vector<int32_t>> seq_ids;  // one list per cell

    bool v_trans = false;

    // One row per cell, rows contiguous. Used for K always, and for V when
    // the cache is not transposed (flash attention on).
    struct layer_rows {
        int32_t  type      = 0;  // ggml_type
        uint64_t row_bytes = 0;
        std::vector<uint8_t> data;  // cell_count * row_bytes
    };
    std::vector<layer_rows> k;
    std::vector<layer_rows> v;

    // Element-major: for each of n_embd elements, all cells' values. Used for
    // V when the cache is transposed (flash attention off).
    struct layer_trans {
        int32_t  type     = 0;
        uint32_t el_bytes = 0;
        uint32_t n_embd   = 0;
        std::vector<uint8_t> data;  // n_embd * cell_count * el_bytes
    };
    std::vector<layer_trans> vt;

    size_t n_layer() const { return k.size(); }
};

// Take a blob apart. Returns false and sets err if the layout does not match.
bool kv_state_parse(const uint8_t * buf, size_t size, kv_state & out, std::string & err);

// Put one back together. The result of parse -> write is byte-identical to
// the input, and a state built from scratch by filling the struct is what a
// context will accept through llama_state_seq_set_data.
std::vector<uint8_t> kv_state_write(const kv_state & st);
