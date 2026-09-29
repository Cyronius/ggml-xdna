// Checks the GGUF-to-chunk weight packer two ways.
//
// Against ggml, here: dequantise the packed pool and compare it to ggml's own
// dequantisation of the same bytes, with the scales narrowed to bfloat16 the
// way a chunk stores them. These must agree exactly. Then compare against
// ggml's unnarrowed dequantisation, which prices what the narrowing costs and
// is the only loss the transform is allowed to have.
//
// Against openflowlm-next's own packer, separately: this writes the raw
// tensor and our pool into a scratch directory, and `tools/check-q4-pack.py`
// runs the reference packer on the same bytes and compares byte for byte.
// Nothing in that repo is read except its packer, and nothing is written.
//
// Traces: HYBRID-Q4-PACK

#include "q4_pack.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _MSC_VER
#define FSEEK64 _fseeki64
#else
#define FSEEK64 fseeko
#endif

namespace {

[[noreturn]] void die(const std::string & what) {
    fprintf(stderr, "error: %s\n", what.c_str());
    exit(1);
}

struct tensor_bytes {
    std::string name;
    int         type = 0;
    int64_t     k = 0, n = 0;   // ne0 (row length), ne1 (rows)
    std::vector<uint8_t> raw;
};

bool read_tensor(FILE * f, gguf_context * gg, ggml_context * gc, int64_t id, tensor_bytes & t) {
    const char * name = gguf_get_tensor_name(gg, id);
    const ggml_tensor * meta = ggml_get_tensor(gc, name);
    if (ggml_n_dims(meta) != 2) return false;

    t.name = name;
    t.type = (int) gguf_get_tensor_type(gg, id);
    t.k    = meta->ne[0];
    t.n    = meta->ne[1];

    const size_t size = gguf_get_tensor_size(gg, id);
    t.raw.resize(size);
    FSEEK64(f, (int64_t) (gguf_get_data_offset(gg) + gguf_get_tensor_offset(gg, id)), SEEK_SET);
    if (fread(t.raw.data(), 1, size, f) != size) die("short read on " + t.name);
    return true;
}

void write_file(const std::string & path, const void * data, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) die("cannot write " + path);
    if (fwrite(data, 1, n, f) != n) die("short write to " + path);
    fclose(f);
}

// ggml's dequantisation of a whole tensor, row by row.
std::vector<float> ggml_dequant(const tensor_bytes & t) {
    const ggml_type ty = (ggml_type) t.type;
    const size_t row_bytes = ggml_row_size(ty, t.k);
    std::vector<float> out((size_t) t.n * t.k);
    const ggml_type_traits * tr = ggml_get_type_traits(ty);
    for (int64_t r = 0; r < t.n; r++) {
        tr->to_float(t.raw.data() + (size_t) r * row_bytes, out.data() + r * t.k, t.k);
    }
    return out;
}

// The same, but with the scale and minimum narrowed to bfloat16 first, which
// is what a chunk can hold. This is what the pool must reproduce exactly.
std::vector<float> narrowed_dequant(const tensor_bytes & t) {
    const bool   q41       = (ggml_type) t.type == GGML_TYPE_Q4_1;
    const size_t blk_bytes = q41 ? 20 : 18;
    const size_t qs_off    = q41 ? 4 : 2;
    const size_t row_bytes = (size_t) (t.k / Q4_BLOCK) * blk_bytes;

    std::vector<float> out((size_t) t.n * t.k);
    for (int64_t r = 0; r < t.n; r++) {
        const uint8_t * row = t.raw.data() + (size_t) r * row_bytes;
        for (int64_t b = 0; b < t.k / Q4_BLOCK; b++) {
            const uint8_t * blk = row + (size_t) b * blk_bytes;
            uint16_t h;
            std::memcpy(&h, blk, 2);
            const float d = ggml_fp16_to_fp32((ggml_fp16_t) h);
            float m;
            if (q41) {
                std::memcpy(&h, blk + 2, 2);
                m = ggml_fp16_to_fp32((ggml_fp16_t) h);
            } else {
                m = -8.0f * d;
            }
            const float dn = q4_bf16_to_f32(q4_f32_to_bf16(d));
            const float mn = q4_bf16_to_f32(q4_f32_to_bf16(m));
            for (int64_t j = 0; j < Q4_BLOCK; j++) {
                const uint8_t byte = blk[qs_off + (j & 15)];
                const uint8_t code = j < 16 ? (uint8_t) (byte & 0x0F) : (uint8_t) (byte >> 4);
                out[r * t.k + b * Q4_BLOCK + j] = (float) code * dn + mn;
            }
        }
    }
    return out;
}

double max_rel(const std::vector<float> & a, const std::vector<float> & b) {
    double worst = 0, scale = 0;
    for (size_t i = 0; i < a.size(); i++) scale = std::max(scale, (double) std::fabs(b[i]));
    for (size_t i = 0; i < a.size(); i++) worst = std::max(worst, (double) std::fabs(a[i] - b[i]));
    return worst / std::max(scale, 1e-30);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, dump_dir;
    int rs = 2;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--dump" && i + 1 < argc) dump_dir = argv[++i];
        else if (a == "--rs"   && i + 1 < argc) rs       = atoi(argv[++i]);
        else if (model_path.empty())            model_path = a;
        else die("unexpected argument " + a);
    }
    if (model_path.empty()) die("usage: test-q4-pack <model.gguf> [--dump dir] [--rs 2|4]");

    ggml_context * gc = nullptr;
    gguf_init_params ip = { /*no_alloc*/ true, &gc };
    gguf_context * gg = gguf_init_from_file(model_path.c_str(), ip);
    if (!gg) die("cannot open " + model_path);
    FILE * f = fopen(model_path.c_str(), "rb");
    if (!f) die("cannot open " + model_path);

    // One tensor of each 4-bit type that tiles, so both paths get exercised.
    std::vector<tensor_bytes> picked;
    bool want_q40 = true, want_q41 = true;
    for (int64_t id = 0; id < gguf_get_n_tensors(gg) && (want_q40 || want_q41); id++) {
        const ggml_type ty = gguf_get_tensor_type(gg, id);
        if (ty == GGML_TYPE_Q4_0 && !want_q40) continue;
        if (ty == GGML_TYPE_Q4_1 && !want_q41) continue;
        if (ty != GGML_TYPE_Q4_0 && ty != GGML_TYPE_Q4_1) continue;

        tensor_bytes t;
        if (!read_tensor(f, gg, gc, id, t)) continue;
        q4_pool_geometry g;
        std::string err;
        if (!q4_pool_geometry_init(t.n, t.k, rs, g, err)) continue;

        if (ty == GGML_TYPE_Q4_0) want_q40 = false; else want_q41 = false;
        picked.push_back(std::move(t));
    }
    if (picked.empty()) die("no Q4_0 or Q4_1 tensor in this file tiles at rs=" + std::to_string(rs));

    int failures = 0;
    auto check = [&](bool ok, const std::string & what) {
        printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
        if (!ok) failures++;
    };

    printf("model: %s   band row split: %d\n\n", model_path.c_str(), rs);

    for (const tensor_bytes & t : picked) {
        const char * tyname = ggml_type_name((ggml_type) t.type);
        q4_pool_geometry g;
        std::string err;
        if (!q4_pool_geometry_init(t.n, t.k, rs, g, err)) die(err);

        printf("%s  %s  K=%lld rows=%lld  -> %lld chunks, %.2f MiB\n",
               t.name.c_str(), tyname, (long long) t.k, (long long) t.n,
               (long long) g.n_chunks, g.bytes() / 1048576.0);

        std::vector<uint8_t> pool(g.bytes());
        if (!q4_pack_pool(t.raw.data(), t.type, g, pool.data(), err)) die(err);

        std::vector<float> from_pool((size_t) t.n * t.k);
        q4_dequant_pool(pool.data(), g, from_pool.data());

        const std::vector<float> want = narrowed_dequant(t);
        const std::vector<float> full = ggml_dequant(t);

        size_t diff = 0;
        for (size_t i = 0; i < want.size(); i++) if (from_pool[i] != want[i]) diff++;

        printf("  pool size matches the layout: %s\n", pool.size() == g.bytes() ? "yes" : "no");
        printf("  narrowing the scale to bfloat16 costs %.2e relative\n", max_rel(want, full));

        check(diff == 0, std::string(tyname) + ": the pool dequantises to exactly the narrowed weights");
        // Q4_0's rule (code - 8) * d equals code * d + (-8 * d) even after
        // narrowing, because scaling by a power of two leaves the mantissa
        // alone. So neither type may lose anything beyond the narrowing.
        check(max_rel(want, full) < 5e-3, std::string(tyname) + ": narrowing is the only loss, and it is small");

        if (!dump_dir.empty()) {
            const std::string stem = dump_dir + "/" + tyname;
            write_file(stem + ".raw", t.raw.data(), t.raw.size());
            write_file(stem + ".pool", pool.data(), pool.size());
            FILE * mf = fopen((stem + ".txt").c_str(), "w");
            if (!mf) die("cannot write " + stem + ".txt");
            fprintf(mf, "name %s\ntype %s\nk %lld\nn %lld\nrs %d\n",
                    t.name.c_str(), tyname, (long long) t.k, (long long) t.n, rs);
            fclose(mf);
            printf("  dumped %s.raw and %s.pool for the cross-check\n", tyname, tyname);
        }
        printf("\n");
    }

    fclose(f);
    ggml_free(gc);
    gguf_free(gg);
    printf("%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
