#include "kv_state.h"

#include <cstdio>
#include <cstring>

namespace {

struct reader {
    const uint8_t * p;
    const uint8_t * end;
    std::string &   err;

    bool take(void * dst, size_t n, const char * what) {
        if ((size_t) (end - p) < n) {
            err = std::string("truncated at ") + what;
            return false;
        }
        std::memcpy(dst, p, n);
        p += n;
        return true;
    }
    template <typename T> bool val(T & v, const char * what) { return take(&v, sizeof(T), what); }
    bool bytes(std::vector<uint8_t> & v, size_t n, const char * what) {
        v.resize(n);
        return n == 0 || take(v.data(), n, what);
    }
};

struct writer {
    std::vector<uint8_t> out;
    template <typename T> void val(const T & v) {
        const uint8_t * b = (const uint8_t *) &v;
        out.insert(out.end(), b, b + sizeof(T));
    }
    void bytes(const std::vector<uint8_t> & v) { out.insert(out.end(), v.begin(), v.end()); }
};

} // namespace

bool kv_state_parse(const uint8_t * buf, size_t size, kv_state & st, std::string & err) {
    st = kv_state{};
    reader r{ buf, buf + size, err };

    uint32_t magic = 0;
    if (!r.val(magic, "magic")) return false;
    if (magic != KV_STATE_MAGIC) {
        char hex[32];
        snprintf(hex, sizeof(hex), "0x%08x", magic);
        err = std::string("bad magic ") + hex;
        return false;
    }
    if (!r.val(st.seq_id, "seq_id")) return false;

    uint32_t n_stream = 0;
    if (!r.val(n_stream, "n_stream")) return false;
    if (n_stream != 1) {
        err = "n_stream is " + std::to_string(n_stream) + ", only 1 is handled";
        return false;
    }

    if (!r.val(st.cell_count, "cell_count")) return false;
    if (st.cell_count == 0) {
        if (r.p != r.end) { err = "trailing bytes after an empty stream"; return false; }
        return true;
    }

    st.pos.resize(st.cell_count);
    st.seq_ids.resize(st.cell_count);
    for (uint32_t i = 0; i < st.cell_count; i++) {
        uint32_t n_seq = 0;
        if (!r.val(st.pos[i], "pos")) return false;
        if (!r.val(n_seq, "n_seq_id")) return false;
        st.seq_ids[i].resize(n_seq);
        for (uint32_t j = 0; j < n_seq; j++) {
            if (!r.val(st.seq_ids[i][j], "seq_id")) return false;
        }
    }

    uint32_t v_trans = 0, n_layer = 0;
    if (!r.val(v_trans, "v_trans")) return false;
    if (!r.val(n_layer, "n_layer")) return false;
    st.v_trans = v_trans != 0;

    st.k.resize(n_layer);
    for (auto & L : st.k) {
        if (!r.val(L.type, "k_type")) return false;
        if (!r.val(L.row_bytes, "k_row_bytes")) return false;
        if (!r.bytes(L.data, (size_t) st.cell_count * L.row_bytes, "k data")) return false;
    }

    if (!st.v_trans) {
        st.v.resize(n_layer);
        for (auto & L : st.v) {
            if (!r.val(L.type, "v_type")) return false;
            if (!r.val(L.row_bytes, "v_row_bytes")) return false;
            if (!r.bytes(L.data, (size_t) st.cell_count * L.row_bytes, "v data")) return false;
        }
    } else {
        st.vt.resize(n_layer);
        for (auto & L : st.vt) {
            if (!r.val(L.type, "v_type")) return false;
            if (!r.val(L.el_bytes, "v_el_bytes")) return false;
            if (!r.val(L.n_embd, "v_n_embd")) return false;
            if (!r.bytes(L.data, (size_t) L.n_embd * st.cell_count * L.el_bytes, "v data")) return false;
        }
    }

    if (r.p != r.end) {
        err = "trailing bytes: " + std::to_string(r.end - r.p);
        return false;
    }
    return true;
}

std::vector<uint8_t> kv_state_write(const kv_state & st) {
    writer w;
    w.val<uint32_t>(KV_STATE_MAGIC);
    w.val<int32_t>(st.seq_id);
    w.val<uint32_t>(1);              // n_stream
    w.val<uint32_t>(st.cell_count);
    if (st.cell_count == 0) {
        return w.out;
    }

    for (uint32_t i = 0; i < st.cell_count; i++) {
        w.val<int32_t>(st.pos[i]);
        w.val<uint32_t>((uint32_t) st.seq_ids[i].size());
        for (int32_t s : st.seq_ids[i]) {
            w.val<int32_t>(s);
        }
    }

    w.val<uint32_t>(st.v_trans ? 1u : 0u);
    w.val<uint32_t>((uint32_t) st.k.size());

    for (const auto & L : st.k) {
        w.val<int32_t>(L.type);
        w.val<uint64_t>(L.row_bytes);
        w.bytes(L.data);
    }
    if (!st.v_trans) {
        for (const auto & L : st.v) {
            w.val<int32_t>(L.type);
            w.val<uint64_t>(L.row_bytes);
            w.bytes(L.data);
        }
    } else {
        for (const auto & L : st.vt) {
            w.val<int32_t>(L.type);
            w.val<uint32_t>(L.el_bytes);
            w.val<uint32_t>(L.n_embd);
            w.bytes(L.data);
        }
    }
    return w.out;
}
