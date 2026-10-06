<#
.SYNOPSIS
Measures how much faster llama.cpp reads a prompt with the NPU add-on than on
the GPU alone, and optionally how far its answers drift, for one or more
models. Prints the table the README's Results section shows.

.DESCRIPTION
Each round is one llama-bench process that tests both the GPU alone and the
GPU with the add-on, so the model loads once and the two measurements sit next
to each other in time. A round's ratio compares that pair; the table gives the
median over the rounds, with the slowest and fastest round.

Results vary from run to run on this kind of machine: the processor, GPU and
NPU share one chip and its power budget. Close other programs first. On
models under about 3B parameters a round takes seconds and the ratio moves
8-14% between rounds, so give those more rounds.

With -Kl it also runs llama-perplexity on the GPU alone and then with the
add-on over the same text, and reports the mean KL divergence between the two
and how often both pick the same most likely next token.

.EXAMPLE
tools\bench.ps1 -Model C:\models\Qwen3-4B-Q4_K_M.gguf

.EXAMPLE
tools\bench.ps1 -Model a.gguf, b.gguf -Rounds 9 -Kl -Text docs.txt
#>
param(
    [Parameter(Mandatory = $true)] [string[]] $Model,
    # rounds per model; each round runs both devices
    [int] $Rounds = 5,
    # prompt length in tokens, read in 2,048-token chunks as npu does
    [int] $Prompt = 2048,
    # also measure answer drift with llama-perplexity
    [switch] $Kl,
    # the text for -Kl: plain text, at least Chunks x 512 tokens
    [string] $Text,
    # 512-token pieces of -Text to compare; 8 is about 4,000 tokens
    [int] $Chunks = 8,
    # the folder with npu.exe and llama.cpp's programs
    [string] $LlamaDir
)

$ErrorActionPreference = 'Stop'

if (-not $LlamaDir) {
    # next to this script (the release zip), else the repo's build of llama.cpp
    $LlamaDir = $PSScriptRoot
    if (-not (Test-Path (Join-Path $LlamaDir 'npu.exe'))) {
        $LlamaDir = Join-Path $PSScriptRoot '..\third_party\llama-b10944'
    }
}
$npu = Join-Path $LlamaDir 'npu.exe'
if (-not (Test-Path $npu)) { throw "no npu.exe in $LlamaDir; pass -LlamaDir" }
if ($Kl -and -not $Text) { throw "-Kl needs -Text, a plain-text file of at least $($Chunks * 512) tokens" }
foreach ($m in $Model) { if (-not (Test-Path $m)) { throw "no such model: $m" } }

# Runs npu.exe with the given arguments and returns its exit code, stdout and
# stderr. Both streams are read at once, so a full pipe never stalls the run.
function Invoke-Npu([string[]] $ArgList) {
    $quoted = $ArgList | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }
    $si = New-Object System.Diagnostics.ProcessStartInfo
    $si.FileName = $npu
    $si.Arguments = $quoted -join ' '
    $si.UseShellExecute = $false
    $si.RedirectStandardOutput = $true
    $si.RedirectStandardError = $true
    $si.CreateNoWindow = $true
    $p = [System.Diagnostics.Process]::Start($si)
    $out = $p.StandardOutput.ReadToEndAsync()
    $err = $p.StandardError.ReadToEndAsync()
    $p.WaitForExit()
    [pscustomobject]@{ Code = $p.ExitCode; Out = $out.Result; Err = $err.Result }
}

function Get-Median([double[]] $v) {
    $s = $v | Sort-Object
    $n = $s.Count
    if ($n % 2) { return $s[($n - 1) / 2] }
    return ($s[$n / 2 - 1] + $s[$n / 2]) / 2
}

# XDNA0 is only offered when the NPU passes its check at start. Without it,
# llama-bench would stop on "invalid device" with no reason given.
$devices = Invoke-Npu @('llama-bench', '--list-devices')
if (($devices.Out + $devices.Err) -notmatch 'XDNA0') {
    $why = ($devices.Out + $devices.Err) -split "`n" | Where-Object { $_ -match 'xdna:' } | Select-Object -First 1
    throw "the NPU add-on isn't available here$(if ($why) { ": $($why.Trim())" })"
}

$rows = @()
foreach ($m in $Model) {
    $name = Split-Path $m -Leaf
    Write-Host "=== $name"
    $ratios = @(); $gpus = @(); $npus = @()
    for ($r = 1; $r -le $Rounds; $r++) {
        $res = Invoke-Npu @('llama-bench', '-m', $m, '-dev', 'Vulkan0,XDNA0/Vulkan0', '-p', "$Prompt", '-n', '0',
                            '-ub', '2048', '-b', '2048', '-r', '3', '-o', 'json')
        $json = [regex]::Match($res.Out, '(?s)\[.*\]').Value
        if (-not $json) {
            Write-Host "   round ${r}: no result (exit $($res.Code))"
            continue
        }
        $tests = $json | ConvertFrom-Json
        if ($tests[0].model_type) { $name = $tests[0].model_type }
        $gpu = ($tests | Where-Object { $_.n_prompt -eq $Prompt -and $_.devices -eq 'Vulkan0' }).avg_ts
        $withNpu = ($tests | Where-Object { $_.n_prompt -eq $Prompt -and $_.devices -eq 'XDNA0/Vulkan0' }).avg_ts
        if (-not $gpu -or -not $withNpu) {
            Write-Host "   round ${r}: incomplete result"
            continue
        }
        $ratios += $withNpu / $gpu; $gpus += $gpu; $npus += $withNpu
        Write-Host ("   round {0}: GPU alone {1,8:N1}  with the add-on {2,8:N1} tokens/s  {3:N2}x" -f $r, $gpu, $withNpu, ($withNpu / $gpu))
    }

    $row = [ordered]@{ Model = $name; Gpu = $null; Ratio = $null; Lo = $null; Hi = $null; Rounds = $ratios.Count;
                       Kld = $null; Top = $null }
    if ($ratios.Count) {
        $row.Gpu = Get-Median $gpus
        $row.Ratio = Get-Median $ratios
        $row.Lo = ($ratios | Measure-Object -Minimum).Minimum
        $row.Hi = ($ratios | Measure-Object -Maximum).Maximum
    }

    if ($Kl) {
        # The GPU's answers first, saved to a file, then the add-on's compared
        # against them. -b 2048 packs four 512-token pieces into each batch, so
        # the add-on gets pieces big enough to take.
        $base = Join-Path $env:TEMP ("xdna-kl-" + [guid]::NewGuid().ToString('N') + '.bin')
        try {
            $common = @('llama-perplexity', '-m', $m, '-f', $Text, '-c', '512', '-b', '2048', '-ub', '2048',
                        '--chunks', "$Chunks", '--kl-divergence-base', $base)
            $g = Invoke-Npu ($common + @('-dev', 'Vulkan0'))
            if ($g.Code -ne 0) { throw "llama-perplexity on the GPU failed (exit $($g.Code))" }
            # -ts 0,1: npu adds it only when it chooses the devices itself, and
            # without it llama.cpp's memory fitting divides by zero on
            # mixture-of-experts models
            $x = Invoke-Npu ($common + @('-dev', 'XDNA0,Vulkan0', '-ts', '0,1', '--kl-divergence'))
            if ($x.Code -ne 0) { throw "llama-perplexity with the add-on failed (exit $($x.Code))" }
            $log = $x.Out + $x.Err
            $kld = [regex]::Match($log, 'Mean\s+KLD:\s+(-?[\d.]+)')
            $top = [regex]::Match($log, 'Same top p:\s+([\d.]+)')
            if ($kld.Success) { $row.Kld = [double]$kld.Groups[1].Value }
            if ($top.Success) { $row.Top = [double]$top.Groups[1].Value }
            Write-Host ("   mean KL divergence {0}, same top token {1}%" -f $row.Kld, $row.Top)
            # bit-identical answers mean the add-on took nothing: a measurement
            # of the GPU against itself, not of the NPU
            if ($row.Kld -eq 0) { Write-Warning "${name}: KL is exactly 0, so the add-on took no work; use a longer -Text" }
        } finally {
            Remove-Item $base -ErrorAction SilentlyContinue
        }
    }
    $rows += [pscustomobject]$row
}

Write-Host ""
$head = '| model | GPU alone, tokens/s | with the add-on | rounds |'
$rule = '|---|---|---|---|'
if ($Kl) { $head += ' mean KL divergence | same top token |'; $rule += '---|---|' }
Write-Host $head
Write-Host $rule
foreach ($row in $rows) {
    $speed = if ($row.Ratio) { '{0:N2}x ({1:N2}-{2:N2})' -f $row.Ratio, $row.Lo, $row.Hi } else { 'no result' }
    $gpu = if ($row.Gpu) { '{0:N0}' -f $row.Gpu } else { '' }
    $line = "| $($row.Model) | $gpu | $speed | $($row.Rounds) |"
    if ($Kl) { $line += " $($row.Kld) | $(if ($row.Top) { "$($row.Top)%" }) |" }
    Write-Host $line
}
