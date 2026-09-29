# Time one built design: samples CPU load and the NPU power mode, then runs bench_bfp16.exe.
#   .\run.ps1 -Design whole_array_bfp -M 512 -K 2048 -N 4096 -Tm 64 -Tk 128 -Tn 64 [-Rne] [-Paired] [-Tag u2] [-Extra 'iters=100 layout=lin']
# Appends the bench's RESULT line plus the CPU load to results.log.
param(
    [Parameter(Mandatory)][string]$Design,
    [Parameter(Mandatory)][int]$M, [Parameter(Mandatory)][int]$K, [Parameter(Mandatory)][int]$N,
    [int]$Tm = 64, [int]$Tk = 64, [int]$Tn = 64, [int]$Cols = 8, [switch]$Rne, [switch]$Paired, [string]$Tag = '', [string]$Extra = ''
)
$here = $PSScriptRoot
$variant = $Design
$args2 = @()
if ($Paired) { $variant += '_paired'; $args2 += 'layout=paired' }
# bench options, space- or comma-separated; later ones win
$args2 += $Extra.Split(' ,'.ToCharArray(), [StringSplitOptions]::RemoveEmptyEntries)
if ($Rne) { $variant += '_rne' }
if ($Tag) { $variant += "_$Tag" }
$dir = Join-Path $here "build\$variant\${M}x${K}x${N}_${Tm}x${Tk}x${Tn}_c${Cols}"
$mode = switch ($Design) { 'whole_array_mixed' { 'mixed' } 'whole_array_bfp_acc' { 'bfpacc' } 'whole_array_bfp_rtp' { 'bfpacc32' } default { 'bfp' } }
if ($Design -eq 'whole_array_bfp_acc' -and $Tag -eq 'f32out') { $mode = 'bfpacc32' }  # --acc 33: fp32 output
$pm = (& C:\Windows\System32\AMD\xrt-smi.exe examine -r platform 2>$null | Select-String 'Power Mode').ToString().Split(':')[-1].Trim()
$cpu0 = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 2).CounterSamples |
    Measure-Object -Property CookedValue -Average | ForEach-Object { $_.Average }
$out = & (Join-Path $here 'out\bench_bfp16.exe') $mode $dir $M $K $N $Tm $Tk $Tn @args2 2>&1
$out | ForEach-Object { Write-Host $_ }
$res = $out | Where-Object { $_ -like 'RESULT*' }
if ($res) {
    $line = "{0} {1} pmode={2} cpu_before={3:N0}% {4}" -f (Get-Date -Format 'HH:mm:ss'), $variant, $pm, $cpu0, $res
    Add-Content -Path (Join-Path $here 'results.log') -Value $line
    Write-Host $line
}
