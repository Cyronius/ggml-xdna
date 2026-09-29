#!/bin/bash
# Re-time the sweep without verification (every design was verified in sweep_run.sh):
# 3 rounds, configs interleaved, CPU load sampled before each run.
# Output: retime.log, one line per run.
cd "$(dirname "$0")"
X=./out/bench_bfp16.exe
cpu() { powershell.exe -NoProfile -Command "[int](Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue"; }
cfgs=()
for M in 256 512 1024; do for KN in "2048 4096" "2048 2048" "2048 12288" "6144 2048"; do set -- $KN; K=$1; N=$2
  cfgs+=("bfp build/whole_array_bfp_lin_paired_rne_u2/${M}x${K}x${N}_64x128x64_c8 $M $K $N 64 128 64 layout=lin")
  [ $M = 512 ] && cfgs+=("bfp build/whole_array_bfp_lin_paired_rne_u2/${M}x${K}x${N}_128x64x64_c8 $M $K $N 128 64 64 layout=lin")
  [ $M = 512 ] && cfgs+=("bfp build/whole_array_bfp/${M}x${K}x${N}_32x256x64_c8 $M $K $N 32 256 64 layout=stock")
  cfgs+=("mixed build/whole_array_mixed_rne/${M}x${K}x${N}_32x256x32_c8 $M $K $N 32 256 32 layout=stock")
done; done
for round in 1 2 3; do
  for c in "${cfgs[@]}"; do
    set -- $c
    load=$(cpu | tr -d '\r')
    r=$($X $c iters=100 verify=0 | grep RESULT)
    echo "$(date +%H:%M:%S) round=$round cpu=$load% dir=$(basename $(dirname $2)) $r" | tee -a retime.log
  done
done
