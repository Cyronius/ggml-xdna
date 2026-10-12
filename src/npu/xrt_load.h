// Linux: the XRT wrapper is a library of its own, loaded at start, so the backend loads without XRT installed.
#pragma once

#include <string>

// Loads XRT, then the wrapper, binding every wrapper function. Empty when all bind, else what failed.
std::string xrt_load();
