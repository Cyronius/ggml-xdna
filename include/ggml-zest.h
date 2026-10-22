#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_XDNA_NAME        "XDNA"   // the backend (registry) name
#define GGML_XDNA_DEVICE_NAME "XDNA0"  // the device, as -dev names it

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna_reg(void);

#ifdef __cplusplus
}
#endif
