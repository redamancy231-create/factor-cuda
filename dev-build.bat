@echo off
rem factor-cuda dev build: configure + build + run selfchecks
rem Requires: VS2026 MSVC, CUDA Toolkit v13.3 (nvcc) -- HOST-LOCKED version, not product support range
rem ASCII only (GBK-safe). Run via: cmd /c dev-build.bat
cd /d "%~dp0"

rem Locate VS via vswhere (no hard-coded path); CUDA via CUDA_PATH env var.
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH goto :err
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 goto :err
set "PATH=%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;%PATH%"
set "CMAKE=%VSPATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

rem arch comes from CMakeLists.txt (sm_89, real+virtual auto) -- do not override
"%CMAKE%" -S . -B build -G Ninja
if errorlevel 1 goto :err
"%CMAKE%" --build build --target phase0_selfcheck poc3_cs_rank_selfcheck poc3_cs_rank_perf poc3_mem_tracker_selfcheck poc3_rolling_ic_selfcheck poc3_rolling_ic_perf poc3_parameter_scan_selfcheck poc3_parameter_scan_perf
if errorlevel 1 goto :err

echo == phase0_selfcheck ==
build\phase0_selfcheck.exe
if errorlevel 1 goto :err
echo == poc3_cs_rank_selfcheck ==
build\poc3_cs_rank_selfcheck.exe
if errorlevel 1 goto :err
echo == poc3_cs_rank_perf ==
build\poc3_cs_rank_perf.exe
if errorlevel 1 goto :err
echo == poc3_mem_tracker_selfcheck ==
build\poc3_mem_tracker_selfcheck.exe
if errorlevel 1 goto :err
echo == poc3_rolling_ic_selfcheck ==
build\poc3_rolling_ic_selfcheck.exe
if errorlevel 1 goto :err
echo == poc3_rolling_ic_perf ==
build\poc3_rolling_ic_perf.exe
exit /b %errorlevel%

:err
echo BUILD_FAILED
exit /b 1
