#!/bin/bash
# Attention insts under the runtime-K xclbin (whole_array_bfp_rtp), per kv head
# and 512-query block: scores M=1024 K=384 N=L, PV M=1024 K=L N=512, L=512..4096.
cd "$(dirname "$0")"
B="powershell.exe -NoProfile -ExecutionPolicy Bypass -File build.ps1 -Design whole_array_bfp_rtp -Tm 128 -Tk 64 -Tn 64"
jl=()
for L in 512 1024 1536 2048 2560 3072 3584 4096; do
  jl+=("-M 1024 -K 384 -N $L")
  jl+=("-M 1024 -K $L -N 512")
done
i=0
for j in "${jl[@]}"; do $B $j 2>&1 | tail -1 & i=$((i+1)); [ $((i % 4)) = 0 ] && wait; done
wait
