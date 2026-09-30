#include "bfp16_insts.h"

#include <algorithm>
#include <set>
#include <utility>

namespace {

constexpr int64_t TM = 128, TK = 64, TN = 64, ROWS = 4, COLS = 8;
constexpr int64_t BLOCK_BYTES = 9;  // eight values sharing an exponent

// Where the toolchain placed each column's streams: the shim column that
// drains C (on its S2MM channel 0), and the shim column and MM2S channel that
// feed A (columns 0-3 only, one per core row) and B. Fixed by the design, not
// by the size.
constexpr int C_SHIM[COLS] = { 3, 2, 4, 1, 5, 6, 7, 0 };
constexpr int A_SHIM[ROWS][2] = { { 0, 0 }, { 2, 0 }, { 4, 0 }, { 6, 0 } };
constexpr int B_SHIM[COLS][2] = { { 3, 0 }, { 3, 1 }, { 2, 1 }, { 4, 1 }, { 5, 0 }, { 5, 1 }, { 6, 1 }, { 1, 0 } };

// Kernel arguments: A, B, C
constexpr uint32_t ARG_A = 0, ARG_B = 1, ARG_C = 2;

// Shim tile registers
constexpr uint32_t BD_BASE = 0x1d000, BD_STRIDE = 0x20;       // buffer descriptors
constexpr uint32_t S2MM0_CTRL = 0x1d200, S2MM0_QUEUE = 0x1d204;
constexpr uint32_t MM2S0_QUEUE = 0x1d214, MM2S1_QUEUE = 0x1d21c;
// Core tile: this dispatch's K/k - 1 and output mode (the design's rtp buffer)
constexpr uint32_t RTP_NMID = 0xfc00, RTP_MODE = 0xfc04;

uint32_t tile_addr(int col, int row, uint32_t off) { return ((uint32_t) col << 25) | ((uint32_t) row << 20) | off; }

struct stream {
    std::vector<uint32_t> w;
    uint32_t ops = 0;
    std::set<int> used[COLS];                 // buffer descriptor IDs in use, per shim column
    std::vector<std::pair<int, int>> group;   // IDs taken since the last wait
    std::vector<int> drains;                  // shim columns of the drains since the last wait

    void put(std::initializer_list<uint32_t> words) { w.insert(w.end(), words); ops++; }
    void write32(uint32_t addr, uint32_t v) { put({ 0, 0, addr, 0, v, 24 }); }

    // One transfer between a kernel argument and a shim DMA channel: a linear
    // buffer descriptor (every transfer this design makes is contiguous), the
    // patch that points it at the argument, and the push onto the queue.
    // A drain also has the channel issue a completion token.
    void task(int col, uint32_t arg, uint32_t off, uint32_t words, uint32_t queue, uint32_t repeat, bool drain) {
        int bd = 0;
        while (used[col].count(bd)) bd++;  // the lowest free ID
        used[col].insert(bd);
        group.push_back({ col, bd });
        const uint32_t a = tile_addr(col, 0, BD_BASE + BD_STRIDE * (uint32_t) bd);
        put({ 1, 0, a, 48, words, off, 0, 0, 0xc0000000, 0x02000000, 0, 0x02000000 });
        put({ 0x81, 48, 0, 0, 0, 0, a + 4, 0, arg, 0, off, 0 });
        if (drain) {
            put({ 3, 0, tile_addr(col, 0, S2MM0_CTRL), 0, 0xf00, 0x1f00, 28 });
            drains.push_back(col);
        }
        write32(tile_addr(col, 0, queue), (drain ? 0x80000000u : 0u) | ((repeat - 1) << 16) | (uint32_t) bd);
    }

    // Waits for the group's drains, then frees its descriptor IDs.
    void finish() {
        for (int col : drains) put({ 0x80, 16, (uint32_t) col << 16, 0x00010100 });
        for (auto [col, bd] : group) used[col].erase(bd);
        group.clear();
        drains.clear();
    }
};

} // namespace

bool bfp16_insts_fits(int64_t M, int64_t K, int64_t N) {
    return M > 0 && M % (TM * ROWS) == 0 && N > 0 && N % (TN * COLS) == 0 && N / (TN * COLS) <= 64 && K % TK == 0 &&
           K >= 2 * TK;
}

std::vector<uint32_t> bfp16_insts(int64_t M, int64_t K, int64_t N, int mode) {
    if (!bfp16_insts_fits(M, K, N) || mode < 0 || mode > 2) return {};
    stream s;
    for (int col = 0; col < COLS; col++)
        for (int row = 2; row < 2 + ROWS; row++) {
            s.write32(tile_addr(col, row, RTP_NMID), (uint32_t) (K / TK - 1));
            s.write32(tile_addr(col, row, RTP_MODE), (uint32_t) mode);
        }

    const int64_t row_blocks = M / (TM * ROWS);                 // blocks of 512 rows
    const int64_t per_col = N / TN / COLS;                      // n-wide output tiles per column
    const int64_t c_per_rb = per_col * TM * ROWS * TN;          // floats one column drains per row block
    const int64_t a_blocks = TM * (K / 8), b_blocks = (N / COLS) * (K / 8);
    // A drain covering two row blocks would stride past the shim's 2^20-word
    // limit when N > 2048, so the design drains one row block at a time there.
    const int64_t tb_max = TM * ROWS * N > (1 << 20) ? 2 : 4;

    for (int64_t tb = 0; tb < (row_blocks + tb_max - 1) / tb_max; tb++) {
        for (int pp = 0; pp < 2; pp++) {
            const int64_t row_base = tb * tb_max + pp * tb_max / 2;
            if (row_base >= row_blocks) break;
            const int64_t rows = std::min(tb_max / 2, row_blocks - row_base);
            for (int col = 0; col < COLS; col++) {
                // C: this column's run of the --c-tiled output
                s.task(C_SHIM[col], ARG_C, (uint32_t) ((col * (M * N / COLS) + row_base * c_per_rb) * 4),
                       (uint32_t) (rows * c_per_rb), S2MM0_QUEUE, 1, true);
                for (int64_t r = 0; r < rows; r++) {
                    // A: a block of m rows, once per output tile of the column
                    if (col < ROWS) {
                        const int64_t t = ((row_base + r) * ROWS + col) % (M / TM);
                        s.task(A_SHIM[col][0], ARG_A, (uint32_t) (t * a_blocks * BLOCK_BYTES),
                               (uint32_t) (a_blocks * BLOCK_BYTES / 4), A_SHIM[col][1] ? MM2S1_QUEUE : MM2S0_QUEUE,
                               (uint32_t) per_col, false);
                    }
                    // B: the column's weight rows
                    s.task(B_SHIM[col][0], ARG_B, (uint32_t) (col * b_blocks * BLOCK_BYTES),
                           (uint32_t) (b_blocks * BLOCK_BYTES / 4), B_SHIM[col][1] ? MM2S1_QUEUE : MM2S0_QUEUE, 1, false);
                }
            }
            if (tb > 0 || pp > 0) s.finish();
        }
    }
    s.finish();

    std::vector<uint32_t> out = { 0x06040100, 0x108, s.ops, (uint32_t) ((s.w.size() + 4) * 4) };
    out.insert(out.end(), s.w.begin(), s.w.end());
    return out;
}
