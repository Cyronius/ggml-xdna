# Makes the release zip from a finished build: llama.cpp's own Windows Vulkan
# release, unmodified, with the add-on, its NPU kernel and the npu launcher
# next to it, plus the licenses and a list of checksums. Run after build.cmd;
# the release workflow does (.github/workflows/release.yml).
#
#   tools\package.ps1 -Version v0.1.0
#
# Everything comes from a fresh download and the build's outputs, never from
# third_party\llama-<tag>, where the build also copies its tests.

param(
    [Parameter(Mandatory)] [string] $Version,
    [string] $Tag   = "b10944",
    [string] $Root  = "",
    [string] $Build = "",
    [string] $Out   = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is slow with it on
if (-not $Root) { $Root = (Join-Path $PSScriptRoot ".." | Resolve-Path).Path }
if (-not $Build) { $Build = Join-Path $Root "build" }
if (-not $Out) { $Out = Join-Path $Root "dist" }

$name = "ggml-xdna-$Version-llama-$Tag-win-x64"
$stage = Join-Path $Out $name
$zip = "$stage.zip"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
if (Test-Path $zip) { Remove-Item -Force $zip }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

# llama.cpp's release, as published
$dl = Join-Path $Out "llama-$Tag-bin-win-vulkan-x64.zip"
if (-not (Test-Path $dl)) {
    Invoke-WebRequest -Uri "https://github.com/ggml-org/llama.cpp/releases/download/$Tag/llama-$Tag-bin-win-vulkan-x64.zip" -OutFile $dl
}
Expand-Archive -Path $dl -DestinationPath $stage

# ours
foreach ($f in "ggml-xdna.dll", "npu.exe") {
    $p = Join-Path $Build $f
    if (-not (Test-Path $p)) { throw "no $f in $($Build); build first" }
    Copy-Item $p $stage
}
Copy-Item (Join-Path $Root "kernels\bfp16_gemm\prebuilt\bfp16_gemm.xclbin") $stage
# the README's measurements, for anyone to rerun; it finds npu.exe next to it
Copy-Item (Join-Path $Root "tools\bench.ps1") $stage
foreach ($f in "README.md", "LICENSE", "NOTICE") { Copy-Item (Join-Path $Root $f) $stage }

# Each part's license. llama.cpp's zip carries only OpenMP's (it stays where
# it is), so llama.cpp's own comes from the same tag's source.
$lic = Join-Path $stage "LICENSES"
New-Item -ItemType Directory -Force -Path $lic | Out-Null
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ggml-org/llama.cpp/$Tag/LICENSE" -OutFile (Join-Path $lic "llama.cpp-LICENSE.txt")
Copy-Item (Join-Path $Root "vendor\xrt\LICENSE") (Join-Path $lic "xrt-LICENSE.txt")
Copy-Item (Join-Path $Root "vendor\xrt\NOTICE") (Join-Path $lic "xrt-NOTICE.txt")
Copy-Item (Join-Path $Root "kernels\bfp16_gemm\LICENSE") (Join-Path $lic "mlir-aie-LICENSE.txt")

# every file's SHA-256, in the format sha256sum -c reads
$sums = Get-ChildItem -Recurse -File $stage | Sort-Object FullName | ForEach-Object {
    $rel = $_.FullName.Substring($stage.Length + 1).Replace('\', '/')
    "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower(), $rel
}
# (LF line endings: with CRLF, sha256sum -c takes the CR as part of each name)
[IO.File]::WriteAllText((Join-Path $stage "SHA256SUMS.txt"), (($sums -join "`n") + "`n"))

Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip
$h = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
[IO.File]::WriteAllText("$zip.sha256", "$h  $name.zip`n")
"{0}  {1:N1} MB  sha256 {2}" -f $zip, ((Get-Item $zip).Length / 1MB), $h
