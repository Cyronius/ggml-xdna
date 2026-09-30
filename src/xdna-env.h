// The backend's settings, from environment variables. An empty value counts
// as unset: `set GGML_XDNA_FOO=` on Windows, or `FOO= cmd` in a shell, is how
// people clear one.
#pragma once

#include <cstdint>
#include <cstdlib>

inline const char * xdna_env(const char * name) {
    const char * s = getenv(name);
    return s && *s ? s : nullptr;
}

inline int64_t xdna_env_int(const char * name, int64_t def) {
    const char * s = xdna_env(name);
    return s ? atoll(s) : def;
}
