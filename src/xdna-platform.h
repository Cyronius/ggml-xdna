// What the backend asks of the operating system, on Windows and Linux.
#pragma once

#include <filesystem>

// Physical memory in bytes: in all, and what's available now.
struct xdna_memory {
    double total = 0, available = 0;
};
xdna_memory xdna_system_memory();

// The folder this library was loaded from; empty if unknown.
std::filesystem::path xdna_module_dir();

// Bytes of memory this process may lock, after raising its own limit as far as the system allows; negative: no limit.
double xdna_raise_lock_limit();
// Lowers that limit for this process, until the next raise.
void xdna_lower_lock_limit(double bytes);
