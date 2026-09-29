#!/bin/bash
# Build the stage-2 shape sweep (the designs sweep_run.sh and retime.sh time), 4 at a time.
# Usage: bash sweep_build.sh
cd "$(dirname "$0")"
B="powershell.exe -NoProfile -ExecutionPolicy Bypass -File build.ps1"
jobs_list=()
for M in 256 512 1024; do
  for KN in "2048 4096" "2048 2048" "2048 12288" "6144 2048"; do
    set -- $KN; K=$1; N=$2
    U="-Paired -Rne -Tag u2 -KFlags -DPAIRED_UNROLL=2"
    jobs_list+=("-Design whole_array_bfp_lin -M $M -K $K -N $N -Tm 64 -Tk 128 -Tn 64 $U")
    [ $M = 512 ] && jobs_list+=("-Design whole_array_bfp_lin -M $M -K $K -N $N -Tm 128 -Tk 64 -Tn 64 $U")
    jobs_list+=("-Design whole_array_bfp -M $M -K $K -N $N -Tm 64 -Tk 128 -Tn 64 $U")
    [ $M = 512 ] && jobs_list+=("-Design whole_array_bfp -M $M -K $K -N $N -Tm 32 -Tk 256 -Tn 64")
    jobs_list+=("-Design whole_array_mixed -M $M -K $K -N $N -Tm 32 -Tk 256 -Tn 32 -Rne")
  done
done
i=0
for j in "${jobs_list[@]}"; do
  $B $j 2>&1 | tail -1 &
  i=$((i+1)); if [ $((i % 4)) = 0 ]; then wait; fi
done
wait
