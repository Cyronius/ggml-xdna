#include "xdna-ref.h"

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

// One weight row dequantised at a time, reused across the whole src1 batch, so
// the dequant cost is amortised over n_tokens instead of paid per dot product.
static void mul_mat_rows(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                         int64_t row_begin, int64_t row_end) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    // src1 broadcasts over src0's higher dims
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const ggml_type_traits * traits = ggml_get_type_traits(src0->type);

    std::vector<float> row(ne00);

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;

            const char * s0_slice = (const char *) src0->data + i02*src0->nb[2] + i03*src0->nb[3];
            const char * s1_slice = (const char *) src1->data + i12*src1->nb[2] + i13*src1->nb[3];
            char       * d_slice  = (char *)       dst->data  + i12*dst->nb[2]  + i13*dst->nb[3];

            for (int64_t i01 = row_begin; i01 < row_end && i01 < ne01; i01++) {
                const char * s0 = s0_slice + i01*src0->nb[1];

                if (src0->type == GGML_TYPE_F32) {
                    memcpy(row.data(), s0, ne00*sizeof(float));
                } else {
                    traits->to_float(s0, row.data(), ne00);
                }

                for (int64_t i11 = 0; i11 < ne11; i11++) {
                    const float * s1 = (const float *) (s1_slice + i11*src1->nb[1]);

                    float acc = 0.0f;
                    for (int64_t k = 0; k < ne00; k++) {
                        acc += row[k] * s1[k];
                    }

                    *(float *) (d_slice + i11*dst->nb[1] + i01*dst->nb[0]) = acc;
                }
            }
        }
    }
}

void xdna_ref_mul_mat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int n_threads) {
    const int64_t ne01 = src0->ne[1];

    if (n_threads < 1) {
        n_threads = 1;
    }
    if ((int64_t) n_threads > ne01) {
        n_threads = (int) ne01;
    }

    if (n_threads == 1) {
        mul_mat_rows(src0, src1, dst, 0, ne01);
        return;
    }

    const int64_t chunk = (ne01 + n_threads - 1) / n_threads;

    std::vector<std::thread> workers;
    workers.reserve(n_threads - 1);

    for (int t = 1; t < n_threads; t++) {
        const int64_t begin = t*chunk;
        if (begin >= ne01) {
            break;
        }
        workers.emplace_back([&, begin] {
            mul_mat_rows(src0, src1, dst, begin, begin + chunk);
        });
    }

    mul_mat_rows(src0, src1, dst, 0, chunk);

    for (auto & w : workers) {
        w.join();
    }
}
