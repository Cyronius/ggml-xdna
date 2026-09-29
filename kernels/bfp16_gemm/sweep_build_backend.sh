#!/bin/bash
# The projection shapes llama.cpp runs that the prototype doesn't (plan:
# .claude/plans/backend-size-aware.md, step 2): llama.cpp multiplies by q, k
# and v separately, and by gate and up separately, where the prototype fused
# them. q (2048 -> 2048), o and down are already built by
# sweep_build_modes.sh; this adds k and v (2048 -> 1024) and gate and up
# (2048 -> 6144), in output mode 0 and mode 1 (16-bit), into the same
# build/whole_array_bfp_rtp_m_ct.
cd "$(dirname "$0")"
B="powershell.exe -NoProfile -ExecutionPolicy Bypass -File build.ps1 -Design whole_array_bfp_rtp -Tm 128 -Tk 64 -Tn 64"
jl=()
for M in 512 1024; do
  for KN in "2048 1024" "2048 6144"; do
    set -- $KN
    jl+=("-Tag m_ct -DesignArgs --c-tiled -M $M -K $1 -N $2")
    jl+=("-Tag m_ct -DesignArgs --c-tiled -M $M -K $1 -N $2 -OutMode 1")
  done
done
i=0
for j in "${jl[@]}"; do eval "$B $j" 2>&1 | grep -E "BUILD_" & i=$((i+1)); [ $((i % 4)) = 0 ] && wait; done
wait
