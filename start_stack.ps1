# One-click start: create venv, install deps, sync instrument list + index, start the web UI.
# Usage (from anywhere):  .\start_stack.ps1 [-Port 8765] [-SyncDaily]
#
# Keep this file pure ASCII: Windows PowerShell 5.1 reads BOM-less files in the
# system code page (GBK on Chinese Windows), so UTF-8 Chinese text here would
# turn into garbage or even parse errors.
param(
    [int]$Port = 8765,
    [switch]$SyncDaily      # also run an incremental daily sync (first full sync takes 1-2 h)
)

$ErrorActionPreference = "Stop"

# The project root is wherever this script lives - no hard-coded paths.
$ProjectPath = $PSScriptRoot
$VenvPython = Join-Path $ProjectPath ".venv\Scripts\python.exe"
Set-Location $ProjectPath

if (-not (Test-Path $VenvPython)) {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "Python not found. Install Python 3.10+ and tick 'Add to PATH'."
        exit 1
    }
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

# Fast no-op when everything is already installed.
Write-Host "Checking dependencies..."
& $VenvPython -m pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed"; exit 1 }

Write-Host "Syncing instrument list and index..."
& $VenvPython -m stack.cli sync --instruments --index

if ($SyncDaily) {
    Write-Host "Incremental daily sync..."
    & $VenvPython -m stack.cli sync --daily
}

Write-Host "Starting UI at http://127.0.0.1:$Port"
& $VenvPython -m stack.cli serve --port $Port
