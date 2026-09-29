#!/bin/bash
# Time fp32 / bf16 partial-sum designs next to the bfp16-C (lin) design, same sitting.
# 3 rounds, configs interleaved; round 1 also verifies (RESULT carries frob/maxrel).
cd "$(dirname "$0")"
X=./out/bench_bfp16.exe
cpu() { powershell.exe -NoProfile -Command "[int](Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue" | tr -d '\r'; }
pm() { C:/Windows/System32/AMD/xrt-smi.exe examine -r platform 2>/dev/null | grep "Power Mode" | awk '{print $4}'; }
cfgs=()
for M in 256 512 1024; do for KN in "2048 4096" "2048 2048" "2048 12288" "6144 2048"; do set -- $KN; K=$1; N=$2; S="$M $K $N"
  cfgs+=("bfpacc build/whole_array_bfp_acc_f32d1/${M}x${K}x${N}_64x128x64_c8 $S 64 128 64 layout=lin")
  [ $M = 512 ] && cfgs+=("bfpacc build/whole_array_bfp_acc_f32d2/${M}x${K}x${N}_64x64x64_c8 $S 64 64 64 layout=lin")
  cfgs+=("bfpacc build/whole_array_bfp_acc_bf16d2/${M}x${K}x${N}_64x128x64_c8 $S 64 128 64 layout=lin")
  [ $M -ge 512 ] && cfgs+=("bfpacc build/whole_array_bfp_acc_bf16d2/${M}x${K}x${N}_128x64x64_c8 $S 128 64 64 layout=lin")
  [ $M -ge 512 ] && cfgs+=("bfpacc32 build/whole_array_bfp_acc_f32out/${M}x${K}x${N}_128x64x64_c8 $S 128 64 64 layout=lin")
  cfgs+=("bfp build/whole_array_bfp_lin_paired_rne_u2/${M}x${K}x${N}_64x128x64_c8 $S 64 128 64 layout=lin")
  [ $M = 512 ] && cfgs+=("bfp build/whole_array_bfp_lin_paired_rne_u2/${M}x${K}x${N}_128x64x64_c8 $S 128 64 64 layout=lin")
done; done
for round in 1 2 3; do
  v=0; [ $round = 1 ] && v=1
  for c in "${cfgs[@]}"; do
    set -- $c
    load=$(cpu)
    r=$($X $c iters=100 verify=$v | grep RESULT)
    echo "$(date +%H:%M:%S) round=$round cpu=$load% pmode=$(pm) dir=$(basename $(dirname $2)) $r" | tee -a retime_acc.log
  done
done
