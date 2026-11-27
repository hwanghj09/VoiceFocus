param([string]$Python = '', [switch]$WithHarmony)
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    function Run([string]$Exe, [string[]]$Arguments) {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$Exe failed (exit $LASTEXITCODE)." }
    }
    foreach ($cmd in @('git','curl.exe','node','npm.cmd')) {
        Get-Command $cmd -ErrorAction Stop | Out-Null
    }
    $nodeVersionText = & node --version
    if ($LASTEXITCODE -ne 0) { throw 'Could not determine Node.js version.' }
    $nodeVersion = [version]($nodeVersionText.Trim().TrimStart('v'))
    if ($nodeVersion -lt [version]'22.12.0') { throw 'Node.js 22.12 or newer is required.' }
    Write-Host "Node.js $nodeVersion OK"
    if (!(Test-Path '.venv\Scripts\python.exe')) {
        if ($Python) { Run $Python @('-m','venv','.venv') }
        else { Run 'py' @('-3.11','-m','venv','.venv') }
    }
    $venv = (Resolve-Path '.venv\Scripts\python.exe').Path
    Run $venv @('-c','import sys,struct; assert (3,11)<=sys.version_info[:2]<=(3,12) and struct.calcsize(''P'')==8, ''64-bit Python 3.11 or 3.12 required''')
    $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
    Run $venv @('-m','pip','install','--index-url','https://download.pytorch.org/whl/cpu','torch==2.7.1','torchaudio==2.7.1')
    & (Join-Path $PSScriptRoot 'setup_model.ps1') -Python $venv -Model spk_emb_100
    & (Join-Path $PSScriptRoot 'setup_model.ps1') -Python $venv -Model spk_emb_causal_100
    if ($WithHarmony) {
        & (Join-Path $PSScriptRoot 'setup_harmony.ps1') -Python $venv
    }
    Run $venv @('-m','pip','check')
    Run $venv @('-c','from backend.app.main import app; import torch, torchaudio; print(''Backend imports OK'')')
    Run 'npm.cmd' @('--prefix','frontend','ci','--no-audit','--no-fund')
    Run 'npm.cmd' @('--prefix','frontend','run','build')
    Write-Host 'Setup complete. Run scripts/serve_demo.ps1.'
} finally { Pop-Location }
