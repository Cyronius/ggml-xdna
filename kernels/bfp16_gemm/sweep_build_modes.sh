#!/bin/bash
# Every shape the prefill uses, under whole_array_bfp_rtp with output modes
# (one core program for all): projections into build/whole_array_bfp_rtp_m_ct,
# each in mode 0 and in the 16-bit mode the prefill reads it with (q/k/v, o
# and down: 1; gate/up: 2, SiLU(gate) * up); attention in mode 0 into
# build/whole_array_bfp_rtp_m_ctkv, its B read from the shared key/value
# buffers (--b-groups / --b-kfull 4096).
cd "$(dirname "$0")"
B="powershell.exe -NoProfile -ExecutionPolicy Bypass -File build.ps1 -Design whole_array_bfp_rtp -Tm 128 -Tk 64 -Tn 64"
jl=()
for M in 512 1024; do
  for KNm in "2048 4096 1" "2048 2048 1" "2048 12288 2" "6144 2048 1"; do
    set -- $KNm
    jl+=("-Tag m_ct -DesignArgs --c-tiled -M $M -K $1 -N $2")
    jl+=("-Tag m_ct -DesignArgs --c-tiled -M $M -K $1 -N $2 -OutMode $3")
  done
done
for L in 512 1024 1536 2048 2560 3072 3584 4096; do
  jl+=("-Tag m_ctkv -DesignArgs '--c-tiled --b-groups 4096' -M 1024 -K 384 -N $L")
  jl+=("-Tag m_ctkv -DesignArgs '--c-tiled --b-kfull 4096' -M 1024 -K $L -N 512")
done
i=0
for j in "${jl[@]}"; do eval "$B $j" 2>&1 | grep -E "BUILD_" & i=$((i+1)); [ $((i % 4)) = 0 ] && wait; done
wait
