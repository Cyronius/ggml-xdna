#!/bin/bash
# Runlist vs single start/wait for attention batches, under the runtime-K xclbin.
# Runs each config until it has 3 results with CPU <= 12% before and after
# (up to 10 tries); every line goes to runlist.log tagged keep/drop.
cd "$(dirname "$0")"
D=build/whole_array_bfp_rtp; X=$(cygpath -w "$PWD/$D/512x2048x4096_128x64x64_c8/final.xclbin")
cpu() { powershell.exe -NoProfile -Command "[int](Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue" | tr -d '\r'; }
S1=$D/1024x384x1024_128x64x64_c8:1024:384:1024:16
P1=$D/1024x1024x512_128x64x64_c8:1024:1024:512:16
S4=$D/1024x384x4096_128x64x64_c8:1024:384:4096:16
P4=$D/1024x4096x512_128x64x64_c8:1024:4096:512:16
for cfg in "$S1 $P1" "$S1" "$P1" "$S4 $P4"; do
  got=0; tries=0
  while [ $got -lt 3 ] && [ $tries -lt 10 ]; do
    tries=$((tries+1)); c0=$(cpu); out=$(./out/runlist.exe "$X" 20 $cfg); c1=$(cpu)
    if [ "$c0" -le 12 ] && [ "$c1" -le 12 ]; then got=$((got+1)); tag=keep; else tag=drop; fi
    echo "$tag cpu=$c0/$c1% $(date +%H:%M:%S) $(echo "$out" | tr '\n' '|')" >> runlist.log
    echo "$tag cpu=$c0/$c1% $(echo "$out" | grep -E '^batch|per run|failing' | sed 's/(batch [^)]*)//' | tr '\n' '|')"
  done
done
