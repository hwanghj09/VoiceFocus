$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    $python = if ($env:PYTHON) { $env:PYTHON } else { '.venv\Scripts\python.exe' }
    if (!(Test-Path 'frontend/dist/index.html')) { throw 'Run scripts/setup_demo.ps1 first.' }
    $env:OPENBLAS_NUM_THREADS = if ($env:VOICEFOCUS_BLAS_THREADS) { $env:VOICEFOCUS_BLAS_THREADS } else { '1' }
    $bind = if ($env:VOICEFOCUS_HOST) { $env:VOICEFOCUS_HOST } else { '127.0.0.1' }
    $port = if ($env:VOICEFOCUS_PORT) { $env:VOICEFOCUS_PORT } else { '8000' }
    & $python -m uvicorn backend.app.main:app --host $bind --port $port --workers 1 --no-access-log
    if ($LASTEXITCODE -ne 0) { throw "Server exited with $LASTEXITCODE" }
} finally { Pop-Location }
