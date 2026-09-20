# Project path
$ProjectPath = "C:\Users\WYT\Downloads\stack-main\stack-main"
$VenvPython = "$ProjectPath\.venv\Scripts\python.exe"
$Port = 8765

# Change directory
Set-Location $ProjectPath

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not installed."
    exit 1
}

# Check virtual environment
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment..."
    python -m venv ".venv"
}

# Run sync
Write-Host "Running sync..."
& $VenvPython -m stack.cli sync --instruments --index

# & $VenvPython -m stack.cli sync --daily

# Start server
Write-Host "Starting server..."
& $VenvPython -m stack.cli serve --port $Port
