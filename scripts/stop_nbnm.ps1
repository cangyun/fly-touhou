param([ValidateSet('route','baseline')][string]$Mode = 'route', [string]$ConfigDirectory = '')
$ErrorActionPreference = 'Stop'
$localPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'outputs/nbnm-local.json'
if (-not $ConfigDirectory -and (Test-Path -LiteralPath $localPath)) {
    $ConfigDirectory = (Get-Content -LiteralPath $localPath -Raw | ConvertFrom-Json).config_directory
}
if (-not $ConfigDirectory) { $ConfigDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) 'outputs/nbnm-preparation' }
$configuration = Join-Path $ConfigDirectory ($Mode + '.json')
$specification = Get-Content -LiteralPath $configuration -Raw | ConvertFrom-Json
if (-not (Test-Path -LiteralPath $specification.output)) { throw 'This NBNM run has not been started.' }
Set-Content -LiteralPath (Join-Path $specification.output 'STOP') -Value 'operator stop at next saved boundary' -Encoding utf8
Write-Output 'Stop requested. The current segment or committed optimizer update will be saved.'
