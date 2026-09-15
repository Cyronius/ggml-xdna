// ggml backend for the AMD XDNA2 NPU, prefill only.
//
// The split between prefill and decode is not written here - it falls out of
// ggml_backend_sched_backend_from_buffer, which picks the highest-priority
// backend that supports both the weight's buffer type AND the op. Weights live
// in our host-visible buffer, we claim matmuls only above a batch threshold, so
// decode's n_tokens=1 matmuls fall through to the CPU reading the same bytes.
// One copy of the weights, no hand-written phase handoff.
//
// see .claude/plans/hybrid-npu-prefill-igpu-decode.md

#include "ggml-xdna.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "xdna-ref.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#define XDNA_DESCRIPTION "AMD XDNA2 NPU (prefill)"

// dst->ne[1] tokens for a plain matmul, ne[2] for the MoE indirect form - same
// convention the Vulkan backend uses for its offload threshold.
static int64_t xdna_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:    return op->ne[1];
        case GGML_OP_MUL_MAT_ID: return op->ne[2];
        default:                 return ggml_nrows(op);
    }
}

static int64_t xdna_min_batch() {
    static int64_t v = [] {
        const char * s = getenv("GGML_XDNA_MIN_BATCH");
        return s ? atoll(s) : 32;
    }();
    return v;
}

// A dispatch costs a few ms whatever it computes, so small matmuls are cheaper
// left on the CPU even at a large batch. Break-even is roughly
// dispatch_ms * cpu_gflops; 256 MFLOP is that at ~3 ms and ~50 GFLOP/s.
static int64_t xdna_min_mflop() {
    static int64_t v = [] {
        const char * s = getenv("GGML_XDNA_MIN_MFLOP");
        return s ? atoll(s) : 256;
    }();
    return v;
}

static int xdna_n_threads() {
    static int v = [] {
        const char * s = getenv("GGML_XDNA_N_THREADS");
        if (s) {
            return atoi(s);
        }
        const unsigned hc = std::thread::hardware_concurrency();
        return hc ? (int) hc : 4;
    }();
    return v;
}

//
// buffer
//

struct xdna_buffer {
    void * data;
    size_t size;
};

static void ggml_backend_xdna_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (xdna_buffer *) buffer->context;
    ggml_aligned_free(ctx->data, ctx->size);
    delete ctx;
}

static void * ggml_backend_xdna_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((xdna_buffer *) buffer->context)->data;
}

static void ggml_backend_xdna_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                   uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_xdna_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_xdna_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                                void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static bool ggml_backend_xdna_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_buffer_is_host(src->buffer)) {
        return false;
    }
    memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;
    GGML_UNUSED(buffer);
}

static void ggml_backend_xdna_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (xdna_buffer *) buffer->context;
    memset(ctx->data, value, ctx->size);
}

static const ggml_backend_buffer_i ggml_backend_xdna_buffer_i = {
    /* .free_buffer   = */ ggml_backend_xdna_buffer_free,
    /* .get_base      = */ ggml_backend_xdna_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_xdna_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_xdna_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_xdna_buffer_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_xdna_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_xdna_buffer_clear,
    /* .reset         = */ NULL,
};

//
// buffer type
//

static const char * ggml_backend_xdna_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_XDNA_NAME;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_xdna_buffer_type_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, size);
        return NULL;
    }

    auto * ctx = new xdna_buffer{ data, size };
    return ggml_backend_buffer_init(buft, ggml_backend_xdna_buffer_i, ctx, size);
}

static size_t ggml_backend_xdna_buffer_type_alignment(ggml_backend_buffer_type_t buft) {
    return 64;
    GGML_UNUSED(buft);
}

// Host-visible on purpose: it is what lets the CPU backend claim the same
// weights for decode without a copy (ggml_backend_cpu_device_supports_buft
// accepts any host buft).
static bool ggml_backend_xdna_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_xdna_buffer_type(void);

//
// backend
//

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    return GGML_XDNA_NAME;
    GGML_UNUSED(backend);
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    delete backend;
}

static ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            case GGML_OP_MUL_MAT:
                xdna_ref_mul_mat(node->src[0], node->src[1], node, xdna_n_threads());
                break;
            default:
                GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

static const ggml_backend_i ggml_backend_xdna_i = {
    /* .get_name             = */ ggml_backend_xdna_get_name,
    /* .free                 = */ ggml_backend_xdna_free,
    /* .set_tensor_async     = */ NULL,
    /* .get_tensor_async     = */ NULL,
    /* .set_tensor_2d_async  = */ NULL,
    /* .get_tensor_2d_async  = */ NULL,
    /* .cpy_tensor_async     = */ NULL,
    /* .synchronize          = */ NULL,
    /* .graph_plan_create    = */ NULL,
    /* .graph_plan_free      = */ NULL,
    /* .graph_plan_update    = */ NULL,
    /* .graph_plan_compute   = */ NULL,
    /* .graph_compute        = */ ggml_backend_xdna_graph_compute,
    /* .event_record         = */ NULL,
    /* .event_wait           = */ NULL,
    /* .graph_optimize       = */ NULL,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = { 0x9d, 0x3a, 0x1c, 0x74, 0x2e, 0x58, 0x41, 0xb6,
                              0x8f, 0x0d, 0xc5, 0x93, 0x6a, 0x27, 0xe1, 0x40 };
    return &guid;
}

//
// device
//

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    return GGML_XDNA_NAME;
    GGML_UNUSED(dev);
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    return XDNA_DESCRIPTION;
    GGML_UNUSED(dev);
}

static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // shared DRAM - nothing of our own to report
    *free  = 0;
    *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna_device_get_name(dev);
    props->description = ggml_backend_xdna_device_get_description(dev);
    props->type        = ggml_backend_xdna_device_get_type(dev);
    props->device_id   = NULL;
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_t ggml_backend_xdna_device_init(ggml_backend_dev_t dev, const char * params) {
    auto * backend = new ggml_backend{
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ ggml_backend_xdna_i,
        /* .device  = */ dev,
        /* .context = */ NULL,
    };
    return backend;
    GGML_UNUSED(params);
}

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            break;
        default:
            return false;
    }

    const int64_t batch = xdna_op_batch_size(op);
    if (batch < xdna_min_batch()) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    const int64_t mflop = 2*src0->ne[0]*src0->ne[1]*batch / 1000000;
    if (mflop < xdna_min_mflop()) {
        return false;
    }

    if (op->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }
    // src1 broadcast over src0's higher dims must be a whole multiple
    if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_xdna_buffer_type_name;
    GGML_UNUSED(dev);
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_xdna_buffer_type();
    GGML_UNUSED(dev);
}

static const ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    // deliberately no offload_op: that path copies the weights per op, which is
    // the OllamaAMDNPU failure mode. We only run on weights already in our buffer.
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// reg
//

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_XDNA_NAME;
    GGML_UNUSED(reg);
}

static size_t ggml_backend_xdna_reg_device_count(ggml_backend_reg_t reg) {
    return 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xdna_reg_device_get(ggml_backend_reg_t reg, size_t index);

static const ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_device_get,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ NULL,
    };
    return &reg;
}

static ggml_backend_dev_t ggml_backend_xdna_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_xdna_device_i,
        /* .reg     = */ ggml_backend_xdna_reg(),
        /* .context = */ NULL,
    };
    return &dev;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static ggml_backend_buffer_type_t ggml_backend_xdna_buffer_type(void) {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name       = */ ggml_backend_xdna_buffer_type_name,
            /* .alloc_buffer   = */ ggml_backend_xdna_buffer_type_alloc,
            /* .get_alignment  = */ ggml_backend_xdna_buffer_type_alignment,
            /* .get_max_size   = */ NULL,
            /* .get_alloc_size = */ NULL,
            /* .is_host        = */ ggml_backend_xdna_buffer_type_is_host,
        },
        /* .device  = */ ggml_backend_xdna_reg_device_get(ggml_backend_xdna_reg(), 0),
        /* .context = */ NULL,
    };
    return &buft;
}

static int ggml_backend_xdna_score(void) {
    return 1;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_xdna_score)
