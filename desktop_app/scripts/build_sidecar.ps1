param(
    [string]$Python = "",
    [ValidateSet("x86_64-pc-windows-msvc")]
    [string]$TargetTriple = "x86_64-pc-windows-msvc"
)

$ErrorActionPreference = "Stop"
$DesktopRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $DesktopRoot
$BackendRoot = Join-Path $DesktopRoot "backend"
$OutputDirectory = Join-Path $DesktopRoot "src-tauri\binaries"
$WorkDirectory = Join-Path $DesktopRoot ".sidecar-build"
$ExecutableName = "cubesprite-backend-$TargetTriple.exe"

if ([string]::IsNullOrWhiteSpace($Python) -and $env:CONDA_PREFIX) {
    $Candidate = Join-Path $env:CONDA_PREFIX "python.exe"
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        $Python = $Candidate
    }
}

if ([string]::IsNullOrWhiteSpace($Python) -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Activate the Conda build environment or pass -Python <conda-env>\python.exe. Resolved path: $Python"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $WorkDirectory | Out-Null

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --console `
    --name "cubesprite-backend-$TargetTriple" `
    --paths $BackendRoot `
    --paths $RepoRoot `
    --hidden-import onnxruntime.capi._pybind_state `
    --exclude-module onnx `
    --exclude-module PIL `
    --exclude-module pygame `
    --exclude-module torch `
    --exclude-module torchvision `
    --distpath $OutputDirectory `
    --workpath (Join-Path $WorkDirectory "work") `
    --specpath $WorkDirectory `
    (Join-Path $PSScriptRoot "sidecar_entry.py")

$Executable = Join-Path $OutputDirectory $ExecutableName
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "PyInstaller did not produce the expected sidecar: $Executable"
}

Write-Host "Built Tauri sidecar: $Executable"
