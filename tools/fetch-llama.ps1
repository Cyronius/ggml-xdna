# Fetches the pinned llama.cpp release we build against, plus the matching ggml
# headers, and makes import libs out of the release DLLs.
#
# The release zip ships no .lib files, so we reconstruct them from the export
# tables with dumpbin + lib. Same source tag, so the ABI matches.

param(
    [string] $Tag  = "b10944",
    [string] $Root = (Join-Path $PSScriptRoot ".." | Resolve-Path)
)

$ErrorActionPreference = "Stop"

$tp = Join-Path $Root "third_party"
New-Item -ItemType Directory -Force -Path $tp | Out-Null

$dist = Join-Path $tp "llama-$Tag"
if (-not (Test-Path $dist)) {
    $zip = Join-Path $tp "llama-vulkan.zip"
    $url = "https://github.com/ggml-org/llama.cpp/releases/download/$Tag/llama-$Tag-bin-win-vulkan-x64.zip"
    Write-Host "downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $dist -Force
    Remove-Item $zip
}

$hdr = Join-Path $tp "ggml-$Tag"
if (-not (Test-Path $hdr)) {
    $zip = Join-Path $tp "llama-src-$Tag.zip"
    $url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$Tag.zip"
    Write-Host "downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $zip
    $tmp = Join-Path $tp "src-tmp"
    Expand-Archive -Path $zip -DestinationPath $tmp -Force

    New-Item -ItemType Directory -Force -Path (Join-Path $hdr "include") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $hdr "src")     | Out-Null
    $src = Join-Path $tmp "llama.cpp-$Tag\ggml"
    Copy-Item "$src\include\*.h" (Join-Path $hdr "include")
    foreach ($f in "ggml-backend-impl.h", "ggml-impl.h", "ggml-common.h") {
        Copy-Item "$src\src\$f" (Join-Path $hdr "src")
    }
    Remove-Item $tmp -Recurse -Force
    Remove-Item $zip
}

# Import libs from the release DLLs' export tables.
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found at $vcvars"
}

foreach ($dll in "ggml-base", "ggml") {
    $lib = Join-Path $tp "$dll.lib"
    if (Test-Path $lib) { continue }

    $cmd = Join-Path $env:TEMP "mk-$dll.cmd"
    @"
@echo off
call "$vcvars" >nul
dumpbin /nologo /exports "$dist\$dll.dll" > "$tp\exports-$dll.txt"
"@ | Set-Content -Encoding ascii $cmd
    cmd /c $cmd

    $names = Select-String -Path "$tp\exports-$dll.txt" -Pattern '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)' |
             ForEach-Object { "    " + $_.Matches[0].Groups[1].Value }
    @("LIBRARY $dll", "EXPORTS") + $names | Set-Content -Encoding ascii "$tp\$dll.def"

    @"
@echo off
call "$vcvars" >nul
lib /nologo /def:"$tp\$dll.def" /machine:x64 /out:"$lib"
"@ | Set-Content -Encoding ascii $cmd
    cmd /c $cmd
    Remove-Item $cmd

    Write-Host "built $lib"
}

Write-Host "third_party ready at $tp"
