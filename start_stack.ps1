# 一键启动脚本：建虚拟环境（如需）、装依赖、起服务、开浏览器。
#
# 路径取脚本自身所在目录，不写死——原版硬编码了作者本机的路径，
# 换台机器就跑不了。

$ErrorActionPreference = "Stop"
$ProjectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectPath ".venv\Scripts\python.exe"
$Port = 8765

Set-Location $ProjectPath
Write-Host "项目目录: $ProjectPath"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "未找到 python，请先安装 Python 3.11+" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "创建虚拟环境…"
    python -m venv ".venv"
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
}

# 端口被占用时直接提示，避免 uvicorn 起不来却看不出原因
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "端口 $Port 已被占用（PID $($busy[0].OwningProcess)）。" -ForegroundColor Yellow
    Write-Host "服务可能已在运行，直接打开 http://127.0.0.1:$Port"
    Start-Process "http://127.0.0.1:$Port"
    exit 0
}

Write-Host "启动服务: http://127.0.0.1:$Port"
Start-Process "http://127.0.0.1:$Port"
& $VenvPython -m stack.cli serve --port $Port
