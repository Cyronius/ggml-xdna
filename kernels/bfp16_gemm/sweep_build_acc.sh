#!/bin/bash
# Build whole_array_bfp_acc (fp32 / bf16 partial sums; f32d1/f32d2 = fp32 accumulator + bf16 out,
# bf16d2 = bf16 in place, f32out = fp32 in the single output buffer, fp32 out) across the stage-2 shapes, 4 at a time.
cd "$(dirname "$0")"
B="powershell.exe -NoProfile -ExecutionPolicy Bypass -File build.ps1 -Design whole_array_bfp_acc"
jl=()
for M in 256 512 1024; do
  for KN in "2048 4096" "2048 2048" "2048 12288" "6144 2048"; do
    set -- $KN; K=$1; N=$2; S="-M $M -K $K -N $N"
    jl+=("$S -Tm 64 -Tk 128 -Tn 64 -Tag f32d1 -DesignArgs \"--acc 32 --c-depth 1\"")
    jl+=("$S -Tm 64 -Tk 128 -Tn 64 -Tag bf16d2 -DesignArgs \"--acc 16 --c-depth 2\"")
    [ $M -ge 512 ] && jl+=("$S -Tm 128 -Tk 64 -Tn 64 -Tag bf16d2 -DesignArgs \"--acc 16 --c-depth 2\"")
    [ $M = 512 ] && jl+=("$S -Tm 64 -Tk 64 -Tn 64 -Tag f32d2 -DesignArgs \"--acc 32 --c-depth 2\"")
    [ $M -ge 512 ] && jl+=("$S -Tm 128 -Tk 64 -Tn 64 -Tag f32out -DesignArgs \"--acc 33 --c-depth 1\"")
  done
done
i=0
for j in "${jl[@]}"; do
  eval "$B $j" 2>&1 | tail -1 &
  i=$((i+1)); if [ $((i % 4)) = 0 ]; then wait; fi
done
wait
