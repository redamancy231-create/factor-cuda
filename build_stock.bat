@echo off
rem temp build for stock_corr targets (ASCII only)
rem Locate VS via vswhere (no hard-coded path); CUDA via CUDA_PATH env var.
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH goto :err
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 goto :err
set "PATH=%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;%PATH%"
"%VSPATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --target poc3_stock_corr_selfcheck poc3_stock_corr_perf
if errorlevel 1 goto :err
echo BUILD_OK
exit /b 0
:err
echo BUILD_FAILED
exit /b 1
