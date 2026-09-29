#!/bin/bash
# Time the stage-2 shape sweep, one design at a time; results.log collects RESULT lines.
cd "$(dirname "$0")"
R="powershell.exe -NoProfile -ExecutionPolicy Bypass -File run.ps1"
for M in 256 512 1024; do
  for KN in "2048 4096" "2048 2048" "2048 12288" "6144 2048"; do
    set -- $KN; K=$1; N=$2
    $R -Design whole_array_bfp_lin -M $M -K $K -N $N -Tm 64 -Tk 128 -Tn 64 -Paired -Rne -Tag u2 -Extra "layout=lin iters=100"
    [ $M = 512 ] && $R -Design whole_array_bfp_lin -M $M -K $K -N $N -Tm 128 -Tk 64 -Tn 64 -Paired -Rne -Tag u2 -Extra "layout=lin iters=100"
    $R -Design whole_array_bfp -M $M -K $K -N $N -Tm 64 -Tk 128 -Tn 64 -Paired -Rne -Tag u2 -Extra "iters=100"
    [ $M = 512 ] && $R -Design whole_array_bfp -M $M -K $K -N $N -Tm 32 -Tk 256 -Tn 64 -Extra "iters=100"
    $R -Design whole_array_mixed -M $M -K $K -N $N -Tm 32 -Tk 256 -Tn 32 -Rne -Extra "iters=100"
  done
done
