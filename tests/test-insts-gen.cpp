// Checks the host-made NPU instruction streams against the ones the IRON
// toolchain built (tests/insts/<M>x<K>x<N>_m<mode>.bin, copied from
// kernels/bfp16_gemm/build): every word must match. A wrong buffer
// descriptor can give plausible-looking wrong output, so this is the check.
// To add a size, build it with kernels/bfp16_gemm/build.ps1 (-Design
// whole_array_bfp_rtp -Tm 128 -Tk 64 -Tn 64 -DesignArgs --c-tiled) and copy
// its insts.bin here.
//
// Traces: XDNA-INSTS-GEN

#include "bfp16_insts.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    const std::filesystem::path dir = argc > 1 ? argv[1] : XDNA_INSTS_DIR;
    int n = 0, failures = 0;
    for (const auto & e : std::filesystem::directory_iterator(dir)) {
        long long M, K, N;
        int mode;
        const std::string name = e.path().filename().string();
        if (sscanf(name.c_str(), "%lldx%lldx%lld_m%d.bin", &M, &K, &N, &mode) != 4) continue;
        std::ifstream f(e.path(), std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::vector<uint32_t> want(bytes.size() / 4);
        memcpy(want.data(), bytes.data(), want.size() * 4);
        const std::vector<uint32_t> got = bfp16_insts(M, K, N, mode);
        size_t first = 0;
        while (first < got.size() && first < want.size() && got[first] == want[first]) first++;
        const bool ok = got.size() == want.size() && first == got.size();
        if (ok) printf("PASS %s\n", name.c_str());
        else printf("FAIL %s: %zu words against %zu, first difference at word %zu\n", name.c_str(), got.size(), want.size(), first);
        n++;
        failures += !ok;
    }
    // sizes the design can't take
    for (auto [M, K, N] : { std::array<long long, 3>{ 256, 2048, 1024 }, { 512, 2048, 896 }, { 512, 96, 1024 },
                            { 512, 64, 1024 }, { 512, 2048, 33280 } }) {
        const bool ok = !bfp16_insts_fits(M, K, N) && bfp16_insts(M, K, N, 0).empty();
        printf("%s %lldx%lldx%lld refused\n", ok ? "PASS" : "FAIL", M, K, N);
        failures += !ok;
    }
    if (n == 0) {
        printf("no reference streams in %s\n", dir.string().c_str());
        return 2;
    }
    printf("%d reference streams: %s\n", n, failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
