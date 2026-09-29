@echo off
REM Build bench_bfp16.exe, ctxswitch.exe, mixk.exe, runlist.exe and modes.exe with MSVC against the XRT 2.21 SDK in C:\Xilinx\XRT
REM (same version as the driver's xrt_coreutil.dll).
setlocal
cd /d "%~dp0"
if "%VCVARS64%"=="" set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if "%XRT_ROOT%"=="" set "XRT_ROOT=C:\Xilinx\XRT"
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VCVARS64%" >nul || exit /b 1
if not exist out mkdir out
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS ^
   /I "%XRT_ROOT%\include" /I . bench_bfp16.cpp "%XRT_ROOT%\lib\xrt_coreutil.lib" ^
   /Fe:out\bench_bfp16.exe /Fo:out\ || exit /b 1
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS ^
   /I "%XRT_ROOT%\include" /I . ctxswitch.cpp "%XRT_ROOT%\lib\xrt_coreutil.lib" ^
   /Fe:out\ctxswitch.exe /Fo:out\ || exit /b 1
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS ^
   /I "%XRT_ROOT%\include" /I . mixk.cpp "%XRT_ROOT%\lib\xrt_coreutil.lib" ^
   /Fe:out\mixk.exe /Fo:out\ || exit /b 1
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS ^
   /I "%XRT_ROOT%\include" /I . runlist.cpp "%XRT_ROOT%\lib\xrt_coreutil.lib" ^
   /Fe:out\runlist.exe /Fo:out\ || exit /b 1
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /arch:AVX2 /D_CRT_SECURE_NO_WARNINGS ^
   /I "%XRT_ROOT%\include" /I . /I ..\..\hybrid modes.cpp ..\..\hybrid\bfp16_pack.cpp "%XRT_ROOT%\lib\xrt_coreutil.lib" ^
   /Fe:out\modes.exe /Fo:out\ || exit /b 1
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /arch:AVX2 /D_CRT_SECURE_NO_WARNINGS ^
   /I "%XRT_ROOT%\include" /I . /I ..\..\hybrid userptr.cpp ..\..\hybrid\bfp16_pack.cpp "%XRT_ROOT%\lib\xrt_coreutil.lib" ^
   /Fe:out\userptr.exe /Fo:out\ || exit /b 1
echo OK out\bench_bfp16.exe out\ctxswitch.exe out\mixk.exe out\runlist.exe out\modes.exe out\userptr.exe
