param([string]$ConfigDirectory='', [string]$Python='', [switch]$Watch)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$localPath=Join-Path $projectRoot 'outputs/nbnm-local.json'
if (Test-Path -LiteralPath $localPath) {
    $localSettings=Get-Content -LiteralPath $localPath -Raw | ConvertFrom-Json
    if (-not $ConfigDirectory) { $ConfigDirectory=$localSettings.config_directory }
    if (-not $Python) { $Python=$localSettings.python }
}
if (-not $ConfigDirectory) { $ConfigDirectory=Join-Path $projectRoot 'outputs/nbnm-preparation' }
if (-not $Python) { $Python=Join-Path $projectRoot '.venv/Scripts/python.exe' }
$settings=Get-Content -LiteralPath (Join-Path $ConfigDirectory 'route.json') -Raw | ConvertFrom-Json
do {
    if ($Watch) { Clear-Host }
    & $Python -B -X utf8 (Join-Path $PSScriptRoot 'summarize_nbnm_trial.py') --run $settings.output
    if ($LASTEXITCODE -ne 0) { throw 'NBNM status read failed' }
    if ($Watch) { Start-Sleep -Seconds 5 }
} while ($Watch)
