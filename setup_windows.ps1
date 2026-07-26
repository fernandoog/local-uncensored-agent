# Setup Windows: Python 3.12 venv + CUDA wheel (no source compile)
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

Write-Host ('[setup] project={0}' -f $Root)

function Get-Python312 {
    try {
        $out = & py -3.12 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            return $out.Trim()
        }
    } catch {
        return $null
    }
    return $null
}

$py312 = Get-Python312

if (-not $py312) {
    Write-Host '[setup] Python 3.12 not found - installing via winget...'
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements

    $machinePath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machinePath;$userPath"

    $py312 = Get-Python312
}

if (-not $py312) {
    Write-Host '[error] Python 3.12 still not found. Install it, open a new terminal, and re-run.'
    Write-Host '  winget install -e --id Python.Python.3.12'
    exit 1
}

Write-Host ('[setup] python3.12={0}' -f $py312)

if (Test-Path -LiteralPath '.\.venv') {
    Write-Host '[setup] removing old .venv (may be 3.13)...'
    Remove-Item -Recurse -Force -LiteralPath '.\.venv'
}

& py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host '[error] failed to create .venv'
    exit 1
}

$venvPython = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host '[error] .venv\Scripts\python.exe missing'
    exit 1
}

& $venvPython -m pip install -U pip wheel setuptools
& $venvPython (Join-Path $Root 'install_deps.py')
if ($LASTEXITCODE -ne 0) {
    Write-Host '[error] install_deps.py failed'
    exit 1
}

Write-Host ''
Write-Host '[ok] Activate with:'
Write-Host '  .\.venv\Scripts\Activate.ps1'
Write-Host '  python smoke_hello.py qwen25-1.5b-uncensored-es-q4'
Write-Host '  python main.py'
