#include "npu_projections.h"

#include "qwen3_ref.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace {
using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
} // namespace

npu_projections::context * npu_projections::context_for(int64_t K, int64_t N, std::string & err) {
    const std::string dir = root_ + "/" + std::to_string(BLOCK) + "x" + std::to_string(K) + "x" + std::to_string(N) + "_128x64x64_c8";
    context * c = nullptr;
    for (auto & p : ctxs_) if (p->K == K) c = p.get();
    if (!c) {
        ctxs_.push_back(std::make_unique<context>());
        c = ctxs_.back().get();
        c->K = K;
        if (!c->npu.open(dir + "/final.xclbin", err)) return nullptr;
    }
    if (!c->shape_for_n.count(N)) {
        const int si = c->npu.add_shape(dir + "/insts.bin", BLOCK, K, N, err);
        if (si < 0) return nullptr;
        c->shape_for_n[N] = si;
    }
    return c;
}

bool npu_projections::init(const qwen3_ref & model, const std::string & build_root, int n_threads, std::string & err) {
    root_ = build_root;
    nth_ = n_threads;
    for (int il = 0; il < model.hp().n_layer; il++) {
        for (unsigned which : { qwen3_prefill_opts::QKV, qwen3_prefill_opts::O, qwen3_prefill_opts::GATE_UP,
                                qwen3_prefill_opts::DOWN }) {
            int64_t N, K;
            const std::vector<float> w = model.projection_weights(il, which, N, K);
            context * c = context_for(K, N, err);
            if (!c) return false;
            std::vector<uint8_t> packed;
            bfp16_pack_b(w.data(), N, K, tile_, packed, nth_);
            const int wi = c->npu.add_weights(packed, err);
            if (wi < 0) return false;
            w_[{ il, which }] = { c, wi };
        }
    }
    return true;
}

bool npu_projections::operator()(int layer, unsigned which, const float * x, int64_t T, int64_t K, float * y, int64_t N) {
    auto it = w_.find({ layer, which });
    if (it == w_.end()) { err_ = "no resident weights for this projection"; return false; }
    context * c = it->second.ctx;
    if (c->K != K || !c->shape_for_n.count(N)) { err_ = "projection shape does not match the loaded kernel"; return false; }
    const int si = c->shape_for_n[N];

    xblk_.resize((size_t) (BLOCK * K));
    yblk_.resize((size_t) (BLOCK * N));
    for (int64_t t0 = 0; t0 < T; t0 += BLOCK) {
        const int64_t rows = std::min(BLOCK, T - t0);
        const clk::time_point h0 = clk::now();
        std::copy(x + t0 * K, x + (t0 + rows) * K, xblk_.begin());
        std::fill(xblk_.begin() + rows * K, xblk_.end(), 0.0f);
        bfp16_pack_a(xblk_.data(), BLOCK, K, tile_, apk_, nth_);
        if (!c->npu.set_a(si, apk_, err_)) return false;
        host_ms += ms_since(h0);

        double ms = 0;
        if (!c->npu.run(si, it->second.weights, yblk_.data(), err_, &ms)) return false;
        npu_ms += ms;
        dispatches++;

        const clk::time_point h1 = clk::now();
        std::copy(yblk_.begin(), yblk_.begin() + rows * N, y + t0 * N);
        host_ms += ms_since(h1);
    }
    return true;
}
