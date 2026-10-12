#include "xdna-platform.h"

#include <algorithm>
#include <fstream>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#endif

#ifdef _WIN32

xdna_memory xdna_system_memory() {
    MEMORYSTATUSEX m = {};
    m.dwLength = sizeof(m);
    if (!GlobalMemoryStatusEx(&m)) return {};
    return { (double) m.ullTotalPhys, (double) m.ullAvailPhys };
}

std::filesystem::path xdna_module_dir() {
    HMODULE h = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR) &xdna_module_dir, &h))
        return {};
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(h, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) return {};
    return std::filesystem::path(std::wstring(buf, n)).parent_path();
}

double xdna_raise_lock_limit() { return -1; }

void xdna_lower_lock_limit(double) {}

#else

// Free memory is MemAvailable, not MemFree: the page cache gives way when asked. Taken from Nulifyer's Linux port.
xdna_memory xdna_system_memory() {
    xdna_memory m;
    struct sysinfo si = {};
    if (sysinfo(&si) == 0) {
        m.total = (double) si.totalram * si.mem_unit;
        m.available = (double) si.freeram * si.mem_unit;
    }
    std::ifstream meminfo("/proc/meminfo");
    std::string key, rest;
    double kb = 0;
    while (meminfo >> key >> kb) {
        std::getline(meminfo, rest);
        if (key == "MemAvailable:") {
            m.available = kb * 1024;
            break;
        }
    }
    // inside a container, its own memory limit (cgroup v2) is the machine
    std::ifstream max_f("/sys/fs/cgroup/memory.max"), cur_f("/sys/fs/cgroup/memory.current");
    double limit = 0, used = 0;
    if (max_f >> limit && cur_f >> used) {
        m.total = std::min(m.total, limit);
        m.available = std::min(m.available, std::max(0.0, limit - used));
    }
    return m;
}

std::filesystem::path xdna_module_dir() {
    Dl_info info = {};
    if (!dladdr((const void *) &xdna_module_dir, &info) || !info.dli_fname) return {};
    return std::filesystem::path(info.dli_fname).parent_path();
}

double xdna_raise_lock_limit() {
    struct rlimit r = {};
    if (getrlimit(RLIMIT_MEMLOCK, &r) != 0) return -1;
    if (r.rlim_cur != r.rlim_max) {
        r.rlim_cur = r.rlim_max;
        setrlimit(RLIMIT_MEMLOCK, &r);
        getrlimit(RLIMIT_MEMLOCK, &r);
    }
    return r.rlim_cur == RLIM_INFINITY ? -1 : (double) r.rlim_cur;
}

void xdna_lower_lock_limit(double bytes) {
    struct rlimit r = {};
    if (getrlimit(RLIMIT_MEMLOCK, &r) != 0) return;
    r.rlim_cur = (rlim_t) bytes;
    setrlimit(RLIMIT_MEMLOCK, &r);
}

#endif
