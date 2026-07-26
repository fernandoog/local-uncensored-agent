# Setup Windows: Python 3.12 venv + CUDA wheel (no source compile)
# Run in PowerShell from the project root:
#   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "[setup] project=$Root"

# Ensure Python 3.12
$py312 = $null
try {
    $py312 = & py -3.12 -c "import sys; print(sys.executable)"
} catch {
    $py312 = $null
}

if (-not $py312) {
    Write-Host "[setup] Python 3.12 not found — installing via winget..."
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    $py312 = & py -3.12 -c "import sys; print(sys.executable)"
}

Write-Host "[setup] python3.12=$py312"

if (Test-Path .\.venv) {
    Write-Host "[setup] removing old .venv (may be 3.13)..."
    Remove-Item -Recurse -Force .\.venv
}

& py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -U pip wheel setuptools
& .\.venv\Scripts\python.exe install_deps.py

Write-Host ""
Write-Host "[ok] Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python smoke_hello.py qwen25-1.5b-uncensored-es-q4"
Write-Host "  python main.py"
