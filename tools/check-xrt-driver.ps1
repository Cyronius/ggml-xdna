# Checks an installed NPU driver against the XRT functions the backend
# imports (vendor/xrt-implib/xrt_coreutil.def): each must be exported, by
# name, from the driver's xrt_coreutil.dll. Run it after a driver update; when
# it passes, record the driver version it prints in the .def.
#
# The backend binds these functions when it starts, so a driver that lacks
# one only turns the backend off, with a log line. This says which, ahead of
# time.
#
# To use a new XRT function, add the decorated name the linker reports as
# unresolved (without its __imp_ prefix) to the .def.

param(
    [string] $Dll  = "$env:SystemRoot\System32\xrt_coreutil.dll",
    [string] $Root = (Join-Path $PSScriptRoot ".." | Resolve-Path)
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Dll)) {
    throw "xrt_coreutil.dll not found at $Dll - install the AMD NPU driver, or pass -Dll"
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = if (Test-Path $vswhere) {
    & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
$vcvars = if ($vs) { Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat" }
if (-not $vcvars -or -not (Test-Path $vcvars)) {
    throw "no Visual Studio with the C++ tools found"
}

$exports = Join-Path $env:TEMP "exports-xrt_coreutil.txt"
$cmd     = Join-Path $env:TEMP "dump-xrt_coreutil.cmd"
@"
@echo off
call "$vcvars" >nul
dumpbin /nologo /exports "$Dll" > "$exports"
"@ | Set-Content -Encoding ascii $cmd
cmd /c $cmd
Remove-Item $cmd

# dumpbin rows are "ordinal hint RVA name"; forwarded entries have no RVA.
$have = @{}
Select-String -Path $exports -Pattern '^\s+\d+\s+[0-9A-F]+\s+(?:[0-9A-F]+\s+)?(\S+)\s*$' |
    ForEach-Object { $have[$_.Matches[0].Groups[1].Value] = $true }
Remove-Item $exports
if ($have.Count -eq 0) { throw "no exports read from $Dll" }

$def  = Join-Path $Root "vendor\xrt-implib\xrt_coreutil.def"
$need = Get-Content $def | Where-Object { $_ -match '^\s{4}\S' } | ForEach-Object { $_.Trim() }
$missing = @($need | Where-Object { -not $have.ContainsKey($_) })

$file = (Get-Item $Dll).VersionInfo.FileVersion
$npu  = Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.DeviceName -cmatch '\bNPU\b' } | Select-Object -First 1
Write-Host "xrt_coreutil.dll $file; $($npu.DeviceName) driver $($npu.DriverVersion)"
if ($missing.Count) {
    Write-Host "missing $($missing.Count) of $($need.Count) functions:"
    $missing | ForEach-Object { Write-Host "    $_" }
    exit 1
}
Write-Host "all $($need.Count) functions the backend imports are there"
