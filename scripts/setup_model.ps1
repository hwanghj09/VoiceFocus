param(
    [ValidateSet('spk_emb_100','spk_emb_causal_100')]
    [string]$Model = 'spk_emb_100',
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    if (!$Python) { $Python = (Resolve-Path '.venv\Scripts\python.exe').Path }
    function Run([string]$Exe, [string[]]$Arguments) {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$Exe failed (exit $LASTEXITCODE)." }
    }
    Run $Python @('-c','import torch, torchaudio; assert torch.__version__.split(''+'')[0] == torchaudio.__version__.split(''+'')[0] == ''2.7.1''')
    Run $Python @('-m','pip','install','-r','backend/requirements.txt','-r','backend/requirements-model.txt')
    Run $Python @('-m','pip','install','ai-edge-litert==2.1.6')
    New-Item -ItemType Directory -Force '.voicefocus/vendor','.voicefocus/models' | Out-Null
    function Checkout($Name,$Url,$Revision) {
        $dir = ".voicefocus/vendor/$Name"
        if (!(Test-Path "$dir/.git")) { Run 'git' @('clone',$Url,$dir) }
        Run 'git' @('-C',$dir,'checkout','--detach',$Revision)
    }
    Checkout 'wesep-real-tse' 'https://github.com/REAL-TSE/wesep-real-tse.git' 'd9ed77d788e9bc8ca08d15854defe404e4593773'
    Checkout 'wespeaker' 'https://github.com/wenet-e2e/wespeaker.git' '8f53b6485d9f88a207bd17e7f8dba899495ec794'
    $weights = '.voicefocus/models/yamnet.tflite'
    if (!(Test-Path $weights)) {
        $part = "$weights.part"
        try {
            Run 'curl.exe' @('--fail','--location','--retry','3','--output',$part,'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite')
            Run $Python @('-c','from pathlib import Path; from sys import argv; from backend.app.models.routing import verified_weights; verified_weights(Path(argv[1]))',$part)
            Move-Item $part $weights
        } finally { Remove-Item $part -ErrorAction SilentlyContinue }
    }
    Run $Python @('-c','from backend.app.models.routing import AudioTagger; AudioTagger()')
    $archive = $env:VOICEFOCUS_CHECKPOINT_ARCHIVE
    if (!$archive) { $archive = '.voicefocus/real-tse-ckpt.zip' }
    if (!(Test-Path $archive)) {
        if ($env:VOICEFOCUS_CHECKPOINT_ARCHIVE) { throw "Checkpoint archive not found: $archive" }
        $part = "$archive.part"
        try {
            Run 'curl.exe' @('--fail','--location','--retry','3','--output',$part,'https://drive.usercontent.google.com/download?id=1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC&export=download&confirm=t')
            Run $Python @('scripts/install_checkpoint.py',$part,$Model)
            Move-Item $part $archive
        } finally { Remove-Item $part -ErrorAction SilentlyContinue }
    } else {
        Run $Python @('scripts/install_checkpoint.py',$archive,$Model)
    }
} finally { Pop-Location }
