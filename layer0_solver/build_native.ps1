$ErrorActionPreference = "Stop"
$source = Join-Path $PSScriptRoot "native\layer0_native.cpp"
$outputDir = Join-Path $PSScriptRoot "build"
$output = Join-Path $outputDir "layer0_native.exe"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
g++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra -o $output $source
if ($LASTEXITCODE -ne 0) {
    throw "Native solver compilation failed with exit code $LASTEXITCODE"
}
Write-Output $output
