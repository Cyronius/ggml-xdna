# Paired prefill checks through an already-running, isolated Lemonade instance.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$BaseUrl = 'http://127.0.0.1:13315/api/v1',
    [ValidateRange(1, 100)][int]$Rounds = 3,
    [string]$Text,
    [ValidateRange(1, 100)][int]$Copies = 8,
    [ValidateRange(1, 4096)][int]$MaxTokens = 48,
    [string]$OutDir
)
$ErrorActionPreference = 'Stop'
if (-not $Text) { $Text = "$PSScriptRoot/../tests/prompts/prose.txt" }
if (-not $OutDir) { $OutDir = "$PSScriptRoot/../third_party/lemonade-results" }
$BaseUrl = $BaseUrl.TrimEnd('/')
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
if (Test-Path -LiteralPath "$OutDir/results.json") {
    throw 'Results already exist. Choose a new -OutDir to preserve the previous run.'
}

function Post($Route, $Body) {
    $json = $Body | ConvertTo-Json -Depth 12
    try {
        Invoke-RestMethod "$BaseUrl/$Route" -Method Post -ContentType 'application/json' `
            -Body ([Text.Encoding]::UTF8.GetBytes($json)) -TimeoutSec 600
    } catch {
        if ($Route -eq 'unload' -and $_.Exception.Response.StatusCode -eq 404) { return }
        throw
    }
}

$passage = Get-Content -LiteralPath $Text -Raw
$prompt = ((1..$Copies | ForEach-Object { "Passage $_`:`n$passage" }) -join "`n") +
    "`nSummarize the passage in one sentence. /no_think"
$request = @{
    model = $Model
    messages = @(@{ role = 'user'; content = $prompt })
    max_tokens = $MaxTokens
    temperature = 0
    stream = $false
    cache_prompt = $false
    chat_template_kwargs = @{ enable_thinking = $false }
}
$request | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 "$OutDir/request.json"
@{
    model = $Model
    rounds = $Rounds
    copies = $Copies
    max_tokens = $MaxTokens
    context = 8192
    batch = 2048
    micro_batch = 2048
    flash_attention = $true
    parallel = 1
    prompt_cache = $false
    thinking = $false
    warmup = 'One full uncached request after each load, before the measured request.'
    text_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Text).Hash.ToLowerInvariant()
    script_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
} | ConvertTo-Json | Set-Content -Encoding UTF8 "$OutDir/settings.json"
$rows = @()
$responses = @()

try {
    foreach ($round in 1..$Rounds) {
        $modes = if ($round % 2) { @('gpu', 'npu') } else { @('npu', 'gpu') }
        foreach ($mode in $modes) {
            Post 'unload' @{ model_name = $Model } | Out-Null
            $device = if ($mode -eq 'npu') { 'XDNA0,Vulkan0' } else { 'Vulkan0' }
            $split = if ($mode -eq 'npu') { '0,1' } else { '1' }
            Post 'load' @{
                model_name = $Model
                ctx_size = 8192
                llamacpp_backend = 'vulkan'
                llamacpp_device = $device
                llamacpp_args = "-ts $split -ub 2048 -b 2048 --flash-attn on --parallel 1"
                merge_args = $false
            } | Out-Null

            # Finish load-time weight conversion and GPU shader warmup before timing.
            Post 'chat/completions' $request | Out-Null
            $response = Post 'chat/completions' $request
            if ($null -eq $response.timings -or $response.timings.prompt_n -lt 2048) {
                throw 'Expected llama.cpp timings and at least 2048 prompt tokens. Increase -Copies if needed.'
            }
            if ($response.timings.cache_n -ne 0 -or
                $response.usage.prompt_tokens_details.cached_tokens -gt 0) {
                throw 'Prompt cache was used; the comparison would not measure full prefill.'
            }
            $responses += [pscustomobject]@{ round = $round; mode = $mode; response = $response }
            $row = [pscustomobject]@{
                round = $round
                mode = $mode
                prompt_tokens = $response.timings.prompt_n
                prompt_ms = $response.timings.prompt_ms
                prompt_tokens_s = $response.timings.prompt_per_second
                decode_tokens_s = $response.timings.predicted_per_second
                fingerprint = $response.system_fingerprint
            }
            $rows += $row
            $rows | Export-Csv -NoTypeInformation -Encoding UTF8 "$OutDir/results.csv"
            $responses | ConvertTo-Json -Depth 15 | Set-Content -Encoding UTF8 "$OutDir/results.json"
            Write-Host ("Round {0}, {1}: {2:N1} prompt tokens/s; {3:N1} decode tokens/s" -f
                $round, $mode, $row.prompt_tokens_s, $row.decode_tokens_s)
        }
    }
} finally {
    Post 'unload' @{ model_name = $Model } | Out-Null
}

$ratios = foreach ($round in 1..$Rounds) {
    $gpu = $rows | Where-Object { $_.round -eq $round -and $_.mode -eq 'gpu' }
    $npu = $rows | Where-Object { $_.round -eq $round -and $_.mode -eq 'npu' }
    if ($gpu.prompt_tokens -ne $npu.prompt_tokens) { throw 'Prompt token counts differ.' }
    $gpu.prompt_ms / $npu.prompt_ms
}
$ordered = @($ratios | Sort-Object)
$median = ($ordered[[int][Math]::Floor(($ordered.Count - 1) / 2)] +
           $ordered[[int][Math]::Floor($ordered.Count / 2)]) / 2
Write-Host ("Median paired prefill speed ratio: {0:N3}x ({1} rounds). Results: {2}" -f
    $median, $Rounds, $OutDir)
Write-Host 'Check the server trace for actual NPU calls. These timings exclude model load and are not client streaming TTFT.'
