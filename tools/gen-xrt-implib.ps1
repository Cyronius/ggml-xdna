# Builds an MSVC import lib for the system xrt_coreutil.dll, so the XRT shim in
# vendor/xrt-shim can link xrt::device / xclbin / hw_context against whatever
# XRT the NPU driver installed.
#
# XRT ships no .lib on Windows, so we reconstruct one from the DLL's export
# table with dumpbin + lib - the same trick tools/fetch-llama.ps1 uses for the
# llama.cpp release DLLs. The exports are mangled C++ names. Only named
# exports go in, and without their ordinals: an ordinal in the .def makes the
# import bind by number, and a driver update that renumbers the exports would
# then call the wrong functions. Bound by name, a missing function fails the
# load instead.

param(
    [string] $Dll  = "$env:SystemRoot\System32\xrt_coreutil.dll",
    [string] $Root = (Join-Path $PSScriptRoot ".." | Resolve-Path)
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Dll)) {
    throw "xrt_coreutil.dll not found at $Dll - install the AMD NPU driver, or pass -Dll"
}

$tp = Join-Path $Root "third_party"
New-Item -ItemType Directory -Force -Path $tp | Out-Null

$lib = Join-Path $tp "xrt_coreutil.lib"
if (Test-Path $lib) {
    Write-Host "$lib already exists"
    exit 0
}

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found at $vcvars"
}

$exports = Join-Path $tp "exports-xrt_coreutil.txt"
$def     = Join-Path $tp "xrt_coreutil.def"
$cmd     = Join-Path $env:TEMP "mk-xrt_coreutil.cmd"

@"
@echo off
call "$vcvars" >nul
dumpbin /nologo /exports "$Dll" > "$exports"
"@ | Set-Content -Encoding ascii $cmd
cmd /c $cmd

# dumpbin rows are "ordinal hint RVA name"; forwarded entries have no RVA.
$names = Select-String -Path $exports -Pattern '^\s+(\d+)\s+[0-9A-F]+\s+(?:[0-9A-F]+\s+)?(\S+)\s*$' |
         ForEach-Object {
             $ord  = $_.Matches[0].Groups[1].Value
             $name = $_.Matches[0].Groups[2].Value
             if ($name -eq "[NONAME]") { return }
             "    $name"
         }
if ($names.Count -eq 0) { throw "no exports parsed out of $exports" }

@("LIBRARY xrt_coreutil", "EXPORTS") + $names | Set-Content -Encoding ascii $def

@"
@echo off
call "$vcvars" >nul
lib /nologo /def:"$def" /machine:x64 /out:"$lib"
"@ | Set-Content -Encoding ascii $cmd
cmd /c $cmd
Remove-Item $cmd

if (-not (Test-Path $lib)) { throw "lib did not produce $lib" }
Write-Host "built $lib from $Dll ($($names.Count) exports)"
