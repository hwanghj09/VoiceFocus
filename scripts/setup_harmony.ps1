param([string]$Python = '')
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    if (!$Python) { $Python = (Resolve-Path '.venv\Scripts\python.exe').Path }
    function Run([string]$Exe,[string[]]$Arguments) {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$Exe failed (exit $LASTEXITCODE)." }
    }
    Run $Python @('-m','pip','install','-r','backend/requirements.txt','-r','backend/requirements-harmony.txt')
    New-Item -ItemType Directory -Force '.voicefocus/vendor','.voicefocus/models' | Out-Null
    $vendor = '.voicefocus/vendor/unmixx'
    if (!(Test-Path "$vendor/.git")) { Run 'git' @('clone','https://github.com/jihoojung0106/unmixx.git',$vendor) }
    Run 'git' @('-C',$vendor,'checkout','--detach','8e750521b5942f4717656cac86a23cf0bd90dea5')
    function InstallEmbedding($Variant,$Weights) {
        if (Test-Path $Weights) { return }
        $part = "$Weights.part"
        try {
            Run 'curl.exe' @('--fail','--location','--retry','3','--output',$part,"https://huggingface.co/BernardoTorres/singer-identity/resolve/55e09cdfcd733b1c01f2b94f316a0280995a909e/$Variant/model.ts")
            Run $Python @('-c','from pathlib import Path; from sys import argv; from backend.app.models.harmony import SINGER_SHA256,VICREG_SHA256,verify_file; verify_file(Path(argv[2]),{''contrastive'':SINGER_SHA256,''vicreg'':VICREG_SHA256}[argv[1]])',$Variant,$part)
            Move-Item $part $Weights
        } finally { Remove-Item $part -ErrorAction SilentlyContinue }
    }
    InstallEmbedding 'contrastive' '.voicefocus/models/singer-identity.ts'
    InstallEmbedding 'vicreg' '.voicefocus/models/singer-identity-vicreg.ts'
    Run $Python @('-c','from backend.app.models.harmony import CONFIG_SHA256,SINGER_SHA256,SOURCE,SPLITTER_SHA256,VICREG_SHA256,VICREG_WEIGHTS,WEIGHTS,verify_file; verify_file(WEIGHTS,SINGER_SHA256); verify_file(VICREG_WEIGHTS,VICREG_SHA256); verify_file(SOURCE/''ckpt''/''best.ckpt'',SPLITTER_SHA256); verify_file(SOURCE/''ckpt''/''conf.yml'',CONFIG_SHA256); import asteroid')
    Write-Host 'Harmony model files verified.'
} finally { Pop-Location }
