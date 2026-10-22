<#
.SYNOPSIS
Starts an isolated Lemonade server with the pinned ggml-zest package.
.EXAMPLE
tools\start-lemonade.ps1 -LemonadeExe C:\Lemonade\bin\LemonadeServer.exe -Model C:\models\Qwen3-4B-Q4_K_M.gguf
#>
[CmdletBinding()]
param(
    [string]$LemonadeExe,
    [string]$Model,
    [string]$LlamaDir,
    [string]$StateDir = "$PSScriptRoot/../third_party/lemonade-instance",
    [ValidateRange(1024, 65535)][int]$Port = 13315,
    [switch]$GpuOnly,
    [switch]$PrepareOnly,
    [switch]$Interactive
)
$ErrorActionPreference = 'Stop'

function Select-LocalFile([string]$Title, [string]$Filter) {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    try {
        $dialog.Title = $Title
        $dialog.Filter = $Filter
        $dialog.CheckFileExists = $true
        if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
            throw 'Selection cancelled. No server was started.'
        }
        return $dialog.FileName
    } finally { $dialog.Dispose() }
}

if ($Interactive) {
    Write-Host 'Zest for Lemonade'
    Write-Host 'Quit your existing Lemonade tray server before starting this one.'
    Write-Host 'Tested with Lemonade 2026.43.3. Choose an existing local GGUF model.'
    if (-not $LemonadeExe) {
        $candidates = @()
        $command = Get-Command LemonadeServer.exe -ErrorAction SilentlyContinue
        if ($command) { $candidates += $command.Source }
        foreach ($base in @($env:ProgramFiles, ${env:ProgramFiles(x86)}, "$env:LOCALAPPDATA/Programs")) {
            if ($base) {
                foreach ($folder in @('Lemonade', 'Lemonade Server', 'lemonade_server')) {
                    $candidates += "$base/$folder/bin/LemonadeServer.exe"
                    $candidates += "$base/$folder/LemonadeServer.exe"
                }
            }
        }
        $LemonadeExe = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
            Select-Object -First 1
        if (-not $LemonadeExe) {
            $LemonadeExe = Select-LocalFile 'Choose LemonadeServer.exe (or portable lemond.exe)' `
                'Lemonade server|LemonadeServer.exe;lemond.exe'
        }
    }
    if (-not $Model) { $Model = Select-LocalFile 'Choose the GGUF model to use in Lemonade' 'GGUF model|*.gguf' }
}
if (-not $LemonadeExe -or -not $Model) {
    throw 'Supply -LemonadeExe and -Model, or use Start-Lemonade.cmd to choose them.'
}
$LemonadeExe = (Resolve-Path -LiteralPath $LemonadeExe).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
if ([IO.Path]::GetFileName($LemonadeExe) -notin @('LemonadeServer.exe', 'lemond.exe')) {
    throw 'Choose LemonadeServer.exe (the server) or portable lemond.exe, rather than the desktop app.'
}
if (-not $PrepareOnly) {
    if ([IO.Path]::GetFileName($LemonadeExe) -eq 'LemonadeServer.exe' -and
        (Get-Process -Name LemonadeServer -ErrorAction SilentlyContinue)) {
        throw 'LemonadeServer is already running. Quit its tray server or use portable lemond.exe for a separate instance.'
    }
    $connection = New-Object Net.Sockets.TcpClient
    try {
        $pending = $connection.ConnectAsync('127.0.0.1', $Port)
        try { $listening = $pending.Wait(500) -and $connection.Connected } catch { $listening = $false }
        if ($listening) { throw "Port $Port is already in use. Stop that instance or choose -Port." }
    } finally { $connection.Dispose() }
}
if ([IO.Path]::GetExtension($Model) -ne '.gguf') { throw '-Model must name a GGUF file.' }
if (-not $LlamaDir) {
    $LlamaDir = "$PSScriptRoot/.."
    if (-not (Test-Path "$LlamaDir/llama-server.exe")) {
        $LlamaDir = "$PSScriptRoot/../third_party/llama-b10944"
    }
}
$LlamaDir = (Resolve-Path -LiteralPath $LlamaDir).Path
foreach ($file in @('llama-server.exe', 'ggml-zest.dll')) {
    if (-not (Test-Path -LiteralPath "$LlamaDir/$file" -PathType Leaf)) { throw "Missing $LlamaDir/$file" }
}
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
$StateDir = (Resolve-Path -LiteralPath $StateDir).Path
$marker = "$StateDir/.zest-launcher"
$legacyMarker = "$StateDir/.ggml-xdna-launcher"
if ((Test-Path "$StateDir/config/config.json") -and
    -not (Test-Path $marker) -and -not (Test-Path $legacyMarker)) {
    throw 'This directory contains an existing configuration. Choose a separate -StateDir.'
}
New-Item -ItemType Directory -Force "$StateDir/cache", "$StateDir/config" | Out-Null
Set-Content -LiteralPath $marker -Value 'Managed by tools/start-lemonade.ps1'

$oldBackend = $env:GGML_BACKEND_PATH
$oldTrace = $env:GGML_XDNA_TRACE
$oldHostOnly = $env:GGML_XDNA_HOST_ONLY
try {
    $env:GGML_BACKEND_PATH = "$LlamaDir/ggml-zest.dll"
    $env:GGML_XDNA_TRACE = '1'
    $env:GGML_XDNA_HOST_ONLY = $null
    # PowerShell 5 treats native stderr as an error record, even on a successful probe.
    $probe = New-Object System.Diagnostics.ProcessStartInfo
    $probe.FileName = "$LlamaDir/llama-server.exe"
    $probe.Arguments = '--list-devices'
    $probe.UseShellExecute = $false
    $probe.CreateNoWindow = $true
    $probe.RedirectStandardOutput = $true
    $probe.RedirectStandardError = $true
    $process = [Diagnostics.Process]::Start($probe)
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $devices = $stdout.Result + $stderr.Result
    Set-Content -LiteralPath "$StateDir/device-probe.log" -Value $devices
    if ($process.ExitCode -ne 0 -or $devices -notmatch '(?m)^\s*Vulkan0:\s') {
        throw "The pinned llama-server could not enumerate a Vulkan GPU. See $StateDir/device-probe.log"
    }
    $useNpu = -not $GpuOnly -and $devices -match '(?m)^\s*XDNA0:\s'
    $device = if ($useNpu) { 'XDNA0,Vulkan0' } else { 'Vulkan0' }
    $split = if ($useNpu) { '0,1' } else { '1' }
    $modelId = 'extra.' + [IO.Path]::GetFileNameWithoutExtension($Model)
    $config = @{
        config_version = 2
        host = '127.0.0.1'
        port = $Port
        broadcast = $false
        offline = $true
        no_fetch_executables = $true
        log_level = 'debug'
        log_file = "$StateDir/server.log"
        models_dir = "$StateDir/models"
        extra_models_dir = [IO.Path]::GetDirectoryName($Model)
        llamacpp = @{ backend = 'vulkan'; vulkan_bin = "$LlamaDir/llama-server.exe" }
    }
    $options = @{}
    $options[$modelId] = @{
        ctx_size = 8192
        llamacpp_backend = 'vulkan'
        llamacpp_device = $device
        llamacpp_args = "-ts $split -ub 2048 -b 2048 --flash-attn on --parallel 1"
        merge_args = $true
    }
    $config | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 "$StateDir/config/config.json"
    $options | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 "$StateDir/config/recipe_options.json"
    Write-Host "Model: $modelId; devices: $device"
    if (-not $useNpu -and -not $GpuOnly) {
        Write-Host "NPU unavailable; using the GPU. Reason: $StateDir/device-probe.log"
    }
    Write-Host "Lemonade desktop Base URL: http://127.0.0.1:$Port"
    Write-Host "In Lemonade App: Settings > Base URL > Save, then choose $modelId."
    Write-Host "Server log: $StateDir/server.log"
    Write-Host 'Leave this window open while chatting. Press Ctrl+C to stop the server.'
    if (-not $PrepareOnly) {
        try {
            $server = Start-Process -FilePath $LemonadeExe -ArgumentList @(
                ('"' + "$StateDir/cache" + '"'), ('"' + "$StateDir/config" + '"'),
                '--port', "$Port", '--host', '127.0.0.1', '--no-broadcast'
            ) -WindowStyle Hidden -PassThru
            $server.WaitForExit()
            if ($server.ExitCode -ne 0) { throw "Lemonade exited with code $($server.ExitCode)" }
        } finally {
            if ($server -and -not $server.HasExited) { $server.Kill(); $server.WaitForExit() }
        }
    }
} finally {
    $env:GGML_BACKEND_PATH = $oldBackend
    $env:GGML_XDNA_TRACE = $oldTrace
    $env:GGML_XDNA_HOST_ONLY = $oldHostOnly
}
