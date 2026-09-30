// The bfp16 packer (src/npu/bfp16_pack.cpp) against the host code the NPU
// kernel was verified with (kernels/bfp16_gemm/bench_bfp16.cpp, now in the
// research-archive tag: floatToBfp16Rne, emit_tile, linA, linB, copied below
// unchanged), byte for byte, for both operands and several tilings. Then
// unpacking must give back exactly the decoded values.
//
// Traces: XDNA-BFP16-PACK

#include "bfp16_pack.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace ref {

// ---- from kernels/bfp16_gemm/bench_bfp16.cpp, unchanged ---------------------
static std::vector<uint8_t> floatToBfp16Rne(int size, const float * x) {
    std::vector<uint8_t> r((size_t) size / 8 * 9);
    for (int b = 0; b < size / 8; b++) {
        unsigned maxe = 0;
        for (int i = 0; i < 8; i++) {
            uint32_t u;
            memcpy(&u, &x[b * 8 + i], 4);
            maxe = std::max(maxe, (u >> 23) & 0xFF);
        }
        r[(size_t) b * 9] = (uint8_t) maxe;
        for (int i = 0; i < 8; i++) {
            double q = std::nearbyint(std::ldexp((double) x[b * 8 + i], 133 - (int) maxe));
            q = std::min(127.0, std::max(-128.0, q));
            r[(size_t) b * 9 + 1 + i] = (uint8_t) (int8_t) q;
        }
    }
    return r;
}
static void emit_tile(const std::vector<uint8_t> & bfp, size_t Wb, size_t r0, size_t c0b, size_t th,
                      size_t twb, bool paired, std::vector<uint8_t> & out) {
    if (paired) {
        for (size_t sp = 0; sp < th; sp += 16)
            for (size_t sx = 0; sx < twb; sx += 9)
                for (size_t half = 0; half < 2; half++)
                    for (size_t i = 0; i < 8; i++)
                        for (size_t j = 0; j < 9; j++) out.push_back(bfp[(r0 + sp + half * 8 + i) * Wb + c0b + sx + j]);
    } else {
        for (size_t sy = 0; sy < th; sy += 8)
            for (size_t sx = 0; sx < twb; sx += 9)
                for (size_t i = 0; i < 8; i++)
                    for (size_t j = 0; j < 9; j++) out.push_back(bfp[(r0 + sy + i) * Wb + c0b + sx + j]);
    }
}
static std::vector<uint8_t> linA(int M, int K, int m, int k, const std::vector<uint8_t> & bfp) {
    std::vector<uint8_t> out;
    out.reserve(bfp.size());
    for (int rb = 0; rb < M / m; rb++)
        for (int kk = 0; kk < K / k; kk++) emit_tile(bfp, K / 8 * 9, (size_t) rb * m, (size_t) kk * k / 8 * 9, m, k / 8 * 9, true, out);
    return out;
}
static std::vector<uint8_t> linB(int N, int K, int n, int k, int cols, const std::vector<uint8_t> & bfp) {
    std::vector<uint8_t> out;
    out.reserve(bfp.size());
    for (int c = 0; c < cols; c++)
        for (int t = 0; t < N / n / cols; t++)
            for (int kk = 0; kk < K / k; kk++)
                emit_tile(bfp, K / 8 * 9, (size_t) (c + cols * t) * n, (size_t) kk * k / 8 * 9, n, k / 8 * 9, true, out);
    return out;
}
// -----------------------------------------------------------------------------

} // namespace ref

// Values spanning many exponents inside each block of eight, including the
// cases that matter for rounding: exact zeros, ties, and maxima that round up
// to +128 and must saturate.
static std::vector<float> make_data(size_t n, uint32_t seed) {
    std::mt19937 eng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> ex(-24, 12), kind(0, 9);
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        switch (kind(eng)) {
            case 0:  x[i] = 0.0f; break;
            case 1:  x[i] = std::ldexp(127.75f, ex(eng)); break;       // rounds up past 127 when it is the block max
            case 2:  x[i] = std::ldexp(0.5f + (float) (eng() % 64), ex(eng)); break;  // a tie at some block exponent
            default: x[i] = std::ldexp(nd(eng), ex(eng)); break;
        }
    }
    return x;
}

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char * what) {
        printf("%s %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) failures++;
    };

    struct shape { int M, K, m, k; };
    for (const shape s : { shape{ 512, 256, 128, 64 }, shape{ 256, 384, 64, 128 }, shape{ 1024, 6144, 128, 64 } }) {
        const std::vector<float> x = make_data((size_t) s.M * s.K, 1 + s.M + s.K);
        const std::vector<uint8_t> want = ref::linA(s.M, s.K, s.m, s.k, ref::floatToBfp16Rne(s.M * s.K, x.data()));
        std::vector<uint8_t> got;
        bfp16_tiling t;
        t.m = s.m; t.k = s.k;
        bfp16_pack_a(x.data(), s.M, s.K, t, got, 8);
        char what[128];
        snprintf(what, sizeof(what), "activations %dx%d, tile m=%d k=%d: identical to the kernel's reference layout", s.M, s.K, s.m, s.k);
        check(got == want, what);

        std::vector<uint8_t> placed(got.size());
        for (int r = 0; r < s.M; r++)
            for (int cb = 0; cb < s.K / 8; cb++)
                bfp16_encode_block(x.data() + (size_t) r * s.K + cb * 8, placed.data() + bfp16_a_offset(r, cb, s.K, t));
        snprintf(what, sizeof(what), "activations %dx%d: blocks encoded at bfp16_a_offset give the same bytes", s.M, s.K);
        check(placed == want, what);
        std::fill(placed.begin(), placed.end(), 0);
        for (int r = 0; r < s.M; r++)
            bfp16_a_row(r, s.K, t, [&](int64_t cb, size_t off) { bfp16_encode_block(x.data() + (size_t) r * s.K + cb * 8, placed.data() + off); });
        snprintf(what, sizeof(what), "activations %dx%d: blocks encoded at bfp16_a_row's offsets give the same bytes", s.M, s.K);
        check(placed == want, what);

        std::vector<float> back((size_t) s.M * s.K);
        bfp16_unpack_a(got.data(), s.M, s.K, t, back.data());
        std::vector<float> dec((size_t) s.M * s.K);
        const std::vector<uint8_t> rows = ref::floatToBfp16Rne(s.M * s.K, x.data());
        for (size_t b = 0; b < dec.size() / 8; b++) bfp16_decode_block(rows.data() + b * 9, dec.data() + b * 8);
        snprintf(what, sizeof(what), "activations %dx%d: unpack gives back the decoded values", s.M, s.K);
        check(memcmp(back.data(), dec.data(), back.size() * 4) == 0, what);
    }

    struct wshape { int N, K, n, k, cols; };
    for (const wshape s : { wshape{ 1024, 256, 64, 64, 8 }, wshape{ 4096, 2048, 64, 64, 8 }, wshape{ 2048, 6144, 64, 128, 8 },
                            wshape{ 512, 128, 32, 64, 4 } }) {
        const std::vector<float> w = make_data((size_t) s.N * s.K, 7 + s.N + s.K);
        const std::vector<uint8_t> want = ref::linB(s.N, s.K, s.n, s.k, s.cols, ref::floatToBfp16Rne(s.N * s.K, w.data()));
        std::vector<uint8_t> got;
        bfp16_tiling t;
        t.n = s.n; t.k = s.k; t.cols = s.cols;
        bfp16_pack_b(w.data(), s.N, s.K, t, got, 8);
        char what[128];
        snprintf(what, sizeof(what), "weights %dx%d, tile n=%d k=%d, %d columns: identical to the kernel's reference layout", s.N, s.K,
                 s.n, s.k, s.cols);
        check(got == want, what);

        std::vector<uint8_t> placed(got.size());
        for (int r = 0; r < s.N; r++)
            for (int cb = 0; cb < s.K / 8; cb++)
                bfp16_encode_block(w.data() + (size_t) r * s.K + cb * 8, placed.data() + bfp16_b_offset(r, cb, s.N, s.K, t));
        snprintf(what, sizeof(what), "weights %dx%d: blocks encoded at bfp16_b_offset give the same bytes", s.N, s.K);
        check(placed == want, what);
        std::fill(placed.begin(), placed.end(), 0);
        for (int r = 0; r < s.N; r++)
            bfp16_b_row(r, s.N, s.K, t, [&](int64_t cb, size_t off) { bfp16_encode_block(w.data() + (size_t) r * s.K + cb * 8, placed.data() + off); });
        snprintf(what, sizeof(what), "weights %dx%d: blocks encoded at bfp16_b_row's offsets give the same bytes", s.N, s.K);
        check(placed == want, what);

        std::vector<float> back((size_t) s.N * s.K), dec((size_t) s.N * s.K);
        bfp16_unpack_b(got.data(), s.N, s.K, t, back.data());
        const std::vector<uint8_t> rows = ref::floatToBfp16Rne(s.N * s.K, w.data());
        for (size_t b = 0; b < dec.size() / 8; b++) bfp16_decode_block(rows.data() + b * 9, dec.data() + b * 8);
        snprintf(what, sizeof(what), "weights %dx%d: unpack gives back the decoded values", s.N, s.K);
        check(memcmp(back.data(), dec.data(), back.size() * 4) == 0, what);
    }

    // The saturation case on its own: a block whose maximum rounds up to +128.
    {
        float blk[8] = { 127.75f, 1.0f, -3.5f, 0.0f, 64.0f, -127.5f, 0.25f, 100.5f };
        for (float & v : blk) v = std::ldexp(v, -6);  // exponent byte 127, so q = value * 64 before rounding
        uint8_t enc[9];
        bfp16_encode_block(blk, enc);
        check(enc[0] == 127 && (int8_t) enc[1] == 127 && (int8_t) enc[6] == -128 && (int8_t) enc[8] == 100,
              "a maximum that rounds up to +128 saturates at 127; -127.5 rounds to -128; 100.5 ties to even");
    }

    // Decoding against plain arithmetic, every exponent byte, mantissas
    // spanning -128..127 (the vector path covers bytes >= 7, the scalar
    // path the rest).
    {
        bool ok = true;
        for (int e = 0; e < 256 && ok; e++)
            for (int base = -128; base < 128 && ok; base += 8) {
                uint8_t blk[9] = { (uint8_t) e };
                for (int i = 0; i < 8; i++) blk[1 + i] = (uint8_t) (int8_t) (base + i);
                float got[8];
                bfp16_decode_block(blk, got);
                for (int i = 0; i < 8; i++) ok = ok && got[i] == (float) std::ldexp((double) (base + i), e - 133);
            }
        check(ok, "decoding matches mantissa * 2^(exponent - 133) for every exponent byte");
    }

    // A block of zeros and negative zeros (the padding rows) takes its own
    // fast path; it must match the reference.
    {
        const float zeros[8] = { 0.0f, -0.0f, 0.0f, 0.0f, -0.0f, 0.0f, 0.0f, 0.0f };
        uint8_t enc[9];
        memset(enc, 0xAB, sizeof(enc));
        bfp16_encode_block(zeros, enc);
        check(ref::floatToBfp16Rne(8, zeros) == std::vector<uint8_t>(enc, enc + 9), "a block of zeros matches the reference");
    }

    printf("%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
