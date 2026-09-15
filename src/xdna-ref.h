#pragma once

#include "ggml.h"

// Reference matmul on the host, used until the NPU kernels land. Same numerics
// path as ggml's own fallback: dequantise src0 rows to f32, then dot.
void xdna_ref_mul_mat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int n_threads);
