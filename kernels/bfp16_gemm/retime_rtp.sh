#!/bin/bash
# Baked-K (whole_array_bfp_acc_f32out, own xclbin per shape) vs runtime-K
# (whole_array_bfp_rtp, every shape under the 512x2048x4096 xclbin).
# Keeps a run only if CPU load was <= 12% both before and after it; collects 3
# such runs per config (gives up after 12 tries). Lines go to retime_rtp.log.
cd "$(dirname "$0")"
X=$(cygpath -w "$PWD/build/whole_array_bfp_rtp/512x2048x4096_128x64x64_c8/final.xclbin")
cpu() { powershell.exe -NoProfile -Command "[int](Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue" | tr -d '\r'; }
pm() { C:/Windows/System32/AMD/xrt-smi.exe examine -r platform 2>/dev/null | grep "Power Mode" | awk '{print $4}'; }
for M in 512 1024; do for KN in "2048 4096" "2048 2048" "2048 12288" "6144 2048"; do set -- $KN; K=$1; N=$2
  for v in baked rtp; do
    if [ $v = baked ]; then d=build/whole_array_bfp_acc_f32out/${M}x${K}x${N}_128x64x64_c8; xa=""; else d=build/whole_array_bfp_rtp/${M}x${K}x${N}_128x64x64_c8; xa="xclbin=$X"; fi
    got=0; tries=0
    while [ $got -lt 3 ] && [ $tries -lt 12 ]; do
      tries=$((tries+1)); c0=$(cpu)
      r=$(./out/bench_bfp16.exe bfpacc32 $d $M $K $N 128 64 64 layout=lin iters=100 verify=0 $xa | grep RESULT)
      c1=$(cpu)
      if [ "$c0" -le 12 ] && [ "$c1" -le 12 ]; then got=$((got+1)); tag=keep; else tag=drop; fi
      echo "$(date +%H:%M:%S) $tag cpu=$c0/$c1% pmode=$(pm) $v $r" | tee -a retime_rtp.log
    done
  done
done; done
