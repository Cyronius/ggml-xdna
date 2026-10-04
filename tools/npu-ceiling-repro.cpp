// npu-ceiling-repro: the NPU's buffers lose their contents, silently, once the
// NPU holds more memory than Windows keeps resident for it.
//
// Seen on a Ryzen AI 9 HX 370 (Strix Point), 96 GB, Windows 11 build 26200,
// NPU driver 32.0.20102.3930, firmware 1.1.2.64, XRT 2.21.0: past about 26 GiB
// of NPU buffers, buffers used earlier come back from the NPU as NaN or wrong
// values, and neither XRT nor the driver reports an error. Found running
// llama.cpp with the ggml-xdna add-on; this program shows it with the NPU
// alone (no llama.cpp, no model files, no GPU).
//
// What it does: it fills the NPU with weight buffers a step at a time and,
// after each step, runs one matmul (the add-on's bfp16 kernel, 512 x 5120 x
// 17408) with every buffer, in the order they were made. All buffers hold the
// same small whole-number weights, except that buffer i's shared exponents
// are raised by i % 8. Its right answer is then exactly buffer 0's times
// 2^(i % 8), so every output is checked bit for bit. Buffer 0's answer is
// first checked against the host. Buffers are made as the add-on makes them
// (xrt::ext::bo, written, synced to the device). Each step prints how much
// shared memory Windows counts for each graphics or compute adapter (the
// "GPU Adapter Memory" counters Task Manager reads).
//
// Usage: npu-ceiling-repro [--start-gib 16] [--step-gib 2] [--max-gib 48]
//                          [--passes 1] [--xclbin path]
//                          [--min-free-gib 6] [--min-disk-gib 15]
// It stops before taking memory the machine doesn't have free, or when the
// system drive runs low (Windows may grow its page file).
// Exit code: 0 nothing damaged; 1 damage with no error reported; 2 it
// couldn't start; 3 the driver reported an error (refused a buffer or failed
// a run) before any damage was seen.

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>

#include "bfp16_insts.h"
#include "bfp16_pack.h"
#include "npu_bfp16.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int64_t M = 512, K = 5120, N = 17408;  // Qwen3.8-27B's feed-forward shape, about 100 MB a buffer
constexpr double GiB = 1024.0 * 1024 * 1024;
const bfp16_tiling TILE;

struct options {
    double start_gib = 16, step_gib = 2, max_gib = 48, min_free_gib = 6, min_disk_gib = 15;
    int passes = 1;
    std::string xclbin;
};

std::filesystem::path exe_dir() {
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return std::filesystem::path(std::wstring(buf, n)).parent_path();
}

double free_memory_gib() {
    MEMORYSTATUSEX m = {};
    m.dwLength = sizeof(m);
    GlobalMemoryStatusEx(&m);
    return m.ullAvailPhys / GiB;
}

double free_disk_gib() {
    ULARGE_INTEGER avail = {};
    wchar_t sysdir[MAX_PATH];
    GetSystemWindowsDirectoryW(sysdir, MAX_PATH);
    const std::wstring root = std::wstring(sysdir).substr(0, 3);  // "C:\"
    GetDiskFreeSpaceExW(root.c_str(), &avail, nullptr, nullptr);
    return avail.QuadPart / GiB;
}

// Windows' own count of the shared memory each graphics or compute adapter
// holds, by counter instance ("luid_0x..._0x..._phys_0").
class adapter_meter {
public:
    ~adapter_meter() {
        if (q_) PdhCloseQuery(q_);
    }
    bool open() {
        return PdhOpenQueryW(nullptr, 0, &q_) == ERROR_SUCCESS &&
               PdhAddEnglishCounterW(q_, L"\\GPU Adapter Memory(*)\\Shared Usage", 0, &c_) == ERROR_SUCCESS &&
               PdhCollectQueryData(q_) == ERROR_SUCCESS;
    }
    std::vector<std::pair<std::wstring, double>> read() {
        std::vector<std::pair<std::wstring, double>> out;
        if (!q_ || PdhCollectQueryData(q_) != ERROR_SUCCESS) return out;
        DWORD bytes = 0, count = 0;
        if (PdhGetFormattedCounterArrayW(c_, PDH_FMT_LARGE, &bytes, &count, nullptr) != PDH_MORE_DATA) return out;
        std::vector<uint8_t> buf(bytes);
        auto * items = (PDH_FMT_COUNTERVALUE_ITEM_W *) buf.data();
        if (PdhGetFormattedCounterArrayW(c_, PDH_FMT_LARGE, &bytes, &count, items) != ERROR_SUCCESS) return out;
        for (DWORD i = 0; i < count; i++) out.emplace_back(items[i].szName, (double) items[i].FmtValue.largeValue);
        return out;
    }

private:
    PDH_HQUERY q_ = nullptr;
    PDH_HCOUNTER c_ = nullptr;
};

double usage_of(const std::vector<std::pair<std::wstring, double>> & u, const std::wstring & name) {
    for (const auto & [n, v] : u)
        if (n == name) return v;
    return 0;
}

struct verdict {
    int64_t wrong = 0, nonfinite = 0;
};

// An output against buffer 0's times 2^d. Whole numbers times a power of two
// are exact in float, so any difference at all is damage.
verdict check(const float * c, const std::vector<float> & base, int d) {
    verdict v;
    const float s = std::ldexp(1.0f, d);
    for (size_t j = 0; j < base.size(); j++) {
        if (!std::isfinite(c[j])) {
            v.wrong++;
            v.nonfinite++;
        } else if (c[j] != base[j] * s) {
            v.wrong++;
        }
    }
    return v;
}

void usage() {
    fprintf(stderr,
            "usage: npu-ceiling-repro [--start-gib 16] [--step-gib 2] [--max-gib 48] [--passes 1]\n"
            "                         [--xclbin path] [--min-free-gib 6] [--min-disk-gib 15]\n");
}

bool parse(int argc, char ** argv, options & o) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (i + 1 >= argc) return false;
        const char * v = argv[++i];
        char * end = nullptr;
        const double x = strtod(v, &end);
        const bool num = end && *end == 0 && x > 0;
        if (a == "--start-gib" && num) o.start_gib = x;
        else if (a == "--step-gib" && num) o.step_gib = x;
        else if (a == "--max-gib" && num) o.max_gib = x;
        else if (a == "--min-free-gib" && num) o.min_free_gib = x;
        else if (a == "--min-disk-gib" && num) o.min_disk_gib = x;
        else if (a == "--passes" && num) o.passes = (int) x;
        else if (a == "--xclbin") o.xclbin = v;
        else return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    options o;
    if (!parse(argc, argv, o)) {
        usage();
        return 2;
    }
    if (o.xclbin.empty()) o.xclbin = (exe_dir() / "bfp16_gemm.xclbin").u8string();
    const int threads = (int) std::max(1u, std::thread::hardware_concurrency());

    MEMORYSTATUSEX m = {};
    m.dwLength = sizeof(m);
    GlobalMemoryStatusEx(&m);
    printf("memory %.1f GiB, %.1f GiB free; system drive %.1f GiB free\n", m.ullTotalPhys / GiB, free_memory_gib(),
           free_disk_gib());

    // Operands: small whole numbers, which the kernel's 8-bit blocks and its
    // float32 sums hold exactly.
    std::vector<int8_t> ai((size_t) (M * K)), wi((size_t) (N * K));
    uint32_t seed = 1;
    auto next = [&](int span) {
        seed = seed * 1664525u + 1013904223u;
        return (int8_t) ((int) (seed >> 24) % span - span / 2);
    };
    for (int8_t & v : ai) v = next(5);  // -2..2
    for (int8_t & v : wi) v = next(7);  // -3..3
    std::vector<uint8_t> pa, pw;
    {
        const std::vector<float> a(ai.begin(), ai.end());
        bfp16_pack_a(a.data(), M, K, TILE, pa, threads);
        const std::vector<float> w(wi.begin(), wi.end());
        bfp16_pack_b(w.data(), N, K, TILE, pw, threads);
    }
    const double buf_gib = pw.size() / GiB;

    auto npu = std::make_unique<npu_bfp16>();
    std::string err;
    if (!npu->open(o.xclbin, err)) {
        fprintf(stderr, "cannot start the NPU: %s\n", err.c_str());
        return 2;
    }
    npu->wait_ms = 30000;
    printf("NPU: %s, kernel %s\n", npu->device_name().c_str(), o.xclbin.c_str());
    const int shape = npu->add_shape(bfp16_insts(M, K, N, 0), M, K, N, err);
    if (shape < 0 || !npu->set_a(shape, pa, err)) {
        fprintf(stderr, "cannot set up the matmul: %s\n", err.c_str());
        return 2;
    }

    adapter_meter meter;
    const bool metered = meter.open();
    const auto before = meter.read();

    struct buffer {
        int handle, d, step;
    };
    std::vector<buffer> bufs;
    std::vector<uint8_t> scratch(pw.size());
    // Buffer i: the packed weights with every block's exponent byte (its
    // first) raised by i % 8.
    auto make = [&](int step) {
        const int d = (int) (bufs.size() % 8);
        memcpy(scratch.data(), pw.data(), pw.size());
        if (d)
            for (size_t off = 0; off < scratch.size(); off += BFP16_BLOCK_BYTES) scratch[off] = (uint8_t) (scratch[off] + d);
        const int h = npu->add_weights(scratch, err);
        if (h < 0) return false;
        bufs.push_back({ h, d, step });
        return true;
    };
    // How many of a buffer's bytes, as the CPU sees them, differ from what was
    // written.
    auto cpu_view = [&](const buffer & b) -> int64_t {
        const uint8_t * p = npu->buffer_map(b.handle);
        if (!p) return -1;
        int64_t diff = 0;
        for (size_t off = 0; off < pw.size(); off++) {
            const uint8_t want = off % BFP16_BLOCK_BYTES == 0 ? (uint8_t) (pw[off] + b.d) : pw[off];
            diff += p[off] != want;
        }
        return diff;
    };
    std::vector<float> c((size_t) (M * N));
    auto run = [&](const buffer & b, double & ms) {
        if (npu->run(shape, b.handle, c.data(), err, &ms)) return true;
        if (npu->timed_out) (void) npu.release();  // still on the NPU: freeing could wait on it
        return false;
    };

    // Buffer 0's answer, checked on the host: rows 0, 255 and 511 in full.
    double ms = 0;
    if (!make(0) || !run(bufs[0], ms)) {
        fprintf(stderr, "first buffer: %s\n", err.c_str());
        return 2;
    }
    const std::vector<float> base = c;
    {
        std::vector<float> row((size_t) N);
        for (int64_t r : { (int64_t) 0, M / 2 - 1, M - 1 }) {
            bfp16_c_row(base.data(), M, N, TILE, r, N, row.data());
            for (int64_t n = 0; n < N; n++) {
                int32_t want = 0;
                for (int64_t k = 0; k < K; k++) want += ai[(size_t) (r * K + k)] * wi[(size_t) (n * K + k)];
                if (row[(size_t) n] != (float) want) {
                    fprintf(stderr, "the kernel is wrong before any memory pressure (row %lld, column %lld: %g, not %d)\n",
                            (long long) r, (long long) n, row[(size_t) n], want);
                    return 2;
                }
            }
        }
    }
    printf("buffer 0 checked against the host; buffers are %.3f GiB each\n\n", buf_gib);

    std::wstring npu_counter;  // the adapter whose shared memory grows with ours
    bool damaged = false, reported = false;
    double target = o.start_gib;
    for (int step = 1; target <= o.max_gib + 1e-9; step++, target += o.step_gib) {
        const double add = std::max(0.0, target - bufs.size() * buf_gib);
        if (free_memory_gib() - add < o.min_free_gib) {
            printf("stopping: %.1f GiB more would leave less than %.1f GiB of memory free\n", add, o.min_free_gib);
            break;
        }
        if (free_disk_gib() < o.min_disk_gib) {
            printf("stopping: under %.1f GiB free on the system drive\n", o.min_disk_gib);
            break;
        }
        const auto t0 = std::chrono::steady_clock::now();
        while ((bufs.size() + 1) * buf_gib <= target + 1e-9) {
            if (!make(step)) {
                printf("the driver refused a buffer at %.1f GiB: %s\n", bufs.size() * buf_gib, err.c_str());
                reported = true;
                goto done;
            }
        }
        for (int pass = 1; pass <= o.passes; pass++) {
            std::vector<std::pair<size_t, verdict>> bad;
            std::vector<double> times;
            for (size_t i = 0; i < bufs.size(); i++) {
                if (!run(bufs[i], ms)) {
                    printf("step %d: the NPU failed a run at %.1f GiB (buffer %zu): %s\n", step, bufs.size() * buf_gib, i,
                           err.c_str());
                    reported = true;
                    goto done;
                }
                times.push_back(ms);
                const verdict v = check(c.data(), base, bufs[i].d);
                if (v.wrong) bad.emplace_back(i, v);
            }
            std::sort(times.begin(), times.end());
            const auto use = meter.read();
            if (npu_counter.empty() && metered) {
                double grew = 0;
                for (const auto & [name, v] : use)
                    if (v - usage_of(before, name) > grew) {
                        grew = v - usage_of(before, name);
                        npu_counter = name;
                    }
            }
            double others = 0;
            for (const auto & [name, v] : use)
                if (name != npu_counter) others += v;
            printf("step %2d, pass %d: %5.1f GiB in %3zu buffers | Windows counts NPU %5.1f GiB, other adapters %5.1f GiB | "
                   "%3zu wrong | run median %.0f ms, slowest %.0f ms | %.0f s\n",
                   step, pass, bufs.size() * buf_gib, bufs.size(), usage_of(use, npu_counter) / GiB, others / GiB, bad.size(),
                   times[times.size() / 2], times.back(),
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            fflush(stdout);
            if (bad.empty()) continue;

            damaged = true;
            printf("\n%zu of %zu buffers gave wrong answers, with no error from XRT or the driver. The first few:\n", bad.size(),
                   bufs.size());
            for (size_t j = 0; j < bad.size() && j < 6; j++) {
                const buffer & b = bufs[bad[j].first];
                const int64_t cpu = cpu_view(b);
                printf("  buffer %zu (made at step %d): %lld of %lld values wrong, %lld of them NaN or infinite; "
                       "its bytes as the CPU sees them: %s\n",
                       bad[j].first, b.step, (long long) bad[j].second.wrong, (long long) base.size(),
                       (long long) bad[j].second.nonfinite,
                       cpu < 0 ? "can't map" : cpu == 0 ? "as written" : (std::to_string(cpu) + " bytes changed").c_str());
            }
            const buffer & first = bufs[bad[0].first];
            if (run(first, ms)) {
                const verdict v = check(c.data(), base, first.d);
                printf("  buffer %zu run again at once: %s\n", bad[0].first,
                       v.wrong ? (std::to_string(v.wrong) + " values wrong").c_str() : "right");
            } else {
                printf("  buffer %zu run again at once: the run failed: %s\n", bad[0].first, err.c_str());
            }
            goto done;
        }
    }
done:
    if (!npu_counter.empty()) printf("\nthe NPU's counter: GPU Adapter Memory(%ls)\\Shared Usage\n", npu_counter.c_str());
    printf("%s\n", damaged    ? "result: buffers came back damaged, with no error reported"
                   : reported ? "result: the driver reported an error; nothing came back damaged before it"
                              : "result: nothing damaged");
    return damaged ? 1 : reported ? 3 : 0;
}
