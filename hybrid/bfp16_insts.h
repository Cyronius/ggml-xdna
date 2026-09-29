// The NPU instruction stream (insts.bin) for one size of the bfp16 matmul,
// made on the host instead of by the IRON toolchain. Plan:
// .claude/plans/any-model-sizes.md.
//
// kernels/bfp16_gemm/designs/whole_array_bfp_rtp.py builds one core program
// that serves every size: its xclbin is the same whatever M, K, N and output
// mode it was built for (only a timestamp, an ID and starting values the
// stream overwrites differ). What changes per size is the runtime sequence:
// each core's K and output mode, then the shim DMA transfers that stream A and
// B in and C out. This reproduces that sequence, as the design lowers it at
// m = 128, k = 64, n = 64 on 4 x 8 cores with --c-tiled, word for word.
//
// Traces: XDNA-INSTS-GEN
#pragma once

#include <cstdint>
#include <vector>

// Whether the design takes this size: M a multiple of 512, N a multiple of
// 512 and at most 32,768, K a multiple of 64 and at least 128.
bool bfp16_insts_fits(int64_t M, int64_t K, int64_t N);

// The stream for C[M x N] = A[M x K] * B^T in output mode 0 (float32), 1
// (bf16) or 2 (SiLU(gate) * up in bf16). Empty if the size doesn't fit.
std::vector<uint32_t> bfp16_insts(int64_t M, int64_t K, int64_t N, int mode);
