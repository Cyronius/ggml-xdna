#!/bin/bash
# Build whole_array_bfp_rtp for every shape. Only insts.bin differs between them;
# any one final.xclbin serves all (bench_bfp16.exe xclbin=..., mixk.exe).
cd "$(dirname "$0")"
B="powershell.exe -NoProfile -ExecutionPolicy Bypass -File build.ps1 -Design whole_array_bfp_rtp -Tm 128 -Tk 64 -Tn 64"
jl=()
for M in 512 1024; do for KN in "2048 4096" "2048 2048" "2048 12288" "6144 2048"; do set -- $KN; jl+=("-M $M -K $1 -N $2"); done; done
for KN in "128 512" "128 1024" "512 512" "1024 512"; do set -- $KN; jl+=("-M 512 -K $1 -N $2"); done
i=0
for j in "${jl[@]}"; do $B $j 2>&1 | tail -1 & i=$((i+1)); [ $((i % 4)) = 0 ] && wait; done
wait
