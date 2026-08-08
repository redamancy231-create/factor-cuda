# factor-cuda helper: build poc3_* target(s) with the dev toolchain.
# Usage: pwsh -NoProfile -File _build_poc3.ps1 <target> [<target> ...]
# ASCII only (GBK-safe). Temp helper, delete after use.
param([Parameter(Mandatory = $true)][string[]]$Targets)

# Locate Visual Studio via vswhere (no hard-coded install path, PII-clean).
# Requires the VS installer component Microsoft.VisualStudio.Component.VC.Tools.x86.x64.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
  Write-Error 'vswhere not found; install Visual Studio 2026 / Build Tools (C++ workload)'
  exit 1
}
$vsPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath) | Select-Object -First 1
if (-not $vsPath) {
  Write-Error 'No Visual Studio with C++ tools found (vswhere)'
  exit 1
}
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
$cmake = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
# vcvars64.bat is a cmd script; wrap the whole chain in cmd /c so the MSVC env
# is set before cmake. Targets are joined into the cmd command line.
# NOTE: do NOT `set PATH=CUDA...;%PATH%` here -- CUDA\v13.3\bin is already on the
# user PATH, and re-prepending it via %PATH% expansion truncates the (very long)
# PATH inside set, dropping the cl.exe directory and breaking nvcc. vcvars sets
# cl; the pre-existing PATH keeps nvcc findable.
$cmd = 'call "' + $vcvars + '" >nul 2>&1 && cd /d "' + $PSScriptRoot + '" && "' + $cmake + '" --build build --target ' + ($Targets -join ' ')
cmd /c $cmd
exit $LASTEXITCODE
