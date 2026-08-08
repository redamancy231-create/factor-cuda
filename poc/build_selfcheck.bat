@echo off
rem Phase 0 self-check build+run script (ASCII only)
cd /d "%~dp0"
rem Locate VS via vswhere (no hard-coded path); CUDA via CUDA_PATH env var.
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH goto :err
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 goto :err
"%CUDA_PATH%\bin\nvcc.exe" -arch=sm_89 -O2 phase0_selfcheck.cu -o phase0_selfcheck.exe
if errorlevel 1 goto :err
phase0_selfcheck.exe
exit /b %errorlevel%
:err
echo BUILD_FAILED
exit /b 1
