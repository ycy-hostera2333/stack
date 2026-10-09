# One-click start: create venv, install deps, sync instrument list + index, start the web UI.
# Usage (from anywhere):  .\start_stack.ps1 [-Port 8765] [-SyncDaily]
# If PowerShell refuses to run scripts:
#   powershell -ExecutionPolicy Bypass -File .\start_stack.ps1
#
# Keep this file pure ASCII: Windows PowerShell 5.1 reads BOM-less files in the
# system code page (GBK on Chinese Windows), so UTF-8 Chinese text here would
# turn into garbage or even parse errors.
#
# No global $ErrorActionPreference = "Stop": in Windows PowerShell 5.1 (ISE, or
# when output is redirected) every stderr line of a native program becomes an
# error record, and "Stop" would abort on pip's upgrade notice or on uvicorn's
# INFO log lines. Native failures are checked through $LASTEXITCODE instead.
param(
    [int]$Port = 8765,
    [switch]$SyncDaily      # also sync daily bars (the very first sync is a full 1-2 h download)
)

# The project root is wherever this script lives - no hard-coded paths.
$ProjectPath = $PSScriptRoot
$VenvPython = Join-Path $ProjectPath ".venv\Scripts\python.exe"
Set-Location $ProjectPath -ErrorAction Stop

# Make Python (and pip) read/write UTF-8 regardless of the Windows code page.
$env:PYTHONUTF8 = "1"

if (-not (Test-Path $VenvPython)) {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "Python not found. Install Python 3.10+ and tick 'Add to PATH'."
        exit 1
    }
    Write-Host "Creating virtual environment..."
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Host "Failed to create .venv"; exit 1 }
}

# Fast no-op when everything is already installed.
Write-Host "Checking dependencies..."
& $VenvPython -m pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    # Offline, or a mirror hiccup: if what we need is already importable, keep going.
    & $VenvPython -c "import fastapi, uvicorn, pandas, numpy, akshare" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed and dependencies are missing."; exit 1 }
    Write-Host "pip install failed, but the dependencies are already installed - continuing."
}

Write-Host "Syncing instrument list and index..."
& $VenvPython -m stack.cli sync --instruments --index

if ($SyncDaily) {
    Write-Host "Syncing daily bars..."
    & $VenvPython -m stack.cli sync --daily
}

Write-Host "Starting UI at http://127.0.0.1:$Port"
& $VenvPython -m stack.cli serve --port $Port
