@echo off
rem Builds ggml-xdna from source and runs its tests, in one command:
rem   build.cmd           fetch llama.cpp, build, test
rem   build.cmd notest    fetch llama.cpp, build
rem Needs Visual Studio 2022 (or its Build Tools) with the C++ tools, which
rem bring CMake and Ninja. The NPU tests run only where the NPU driver is
rem installed; the others need a Vulkan GPU.
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File tools\fetch-llama.ps1
if errorlevel 1 exit /b 1

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo No Visual Studio with the C++ tools found.
    exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1

if not exist build\build.ninja cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build build
if errorlevel 1 exit /b 1

if /i "%~1"=="notest" exit /b 0
rem with the NPU driver: every test but the one expecting no driver; without it, no NPU tests
set "SKIP=npu"
if exist "%SystemRoot%\System32\xrt_coreutil.dll" set "SKIP=nodriver"
ctest --test-dir build -LE %SKIP% --output-on-failure
