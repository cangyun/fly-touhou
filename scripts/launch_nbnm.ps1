param(
    [ValidateSet('route','baseline')][string]$Mode = 'route',
    [string]$ConfigDirectory = '',
    [string]$Resume = '',
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ConfigDirectory) { $ConfigDirectory = Join-Path $projectRoot 'outputs/nbnm-preparation' }
if (-not $Python) { $Python = $env:FLY_TOUHOU_PYTHON }
if (-not $Python) { $Python = Join-Path $projectRoot '.venv/Scripts/python.exe' }
$Python = (Resolve-Path -LiteralPath $Python).Path
$configuration = (Resolve-Path -LiteralPath (Join-Path $ConfigDirectory ($Mode + '.json'))).Path
if ($Resume) { $Resume = (Resolve-Path -LiteralPath $Resume).Path }
$specification = Get-Content -LiteralPath $configuration -Raw | ConvertFrom-Json
& $Python -B -X utf8 (Join-Path $PSScriptRoot 'train_nbnm.py') --config $configuration
if ($LASTEXITCODE -ne 0) { throw 'NBNM configuration is not ready; no training was started.' }
if (-not $Resume -and (Test-Path -LiteralPath $specification.output)) {
    throw 'This experiment output already exists. Use -Resume with its latest-training.pt, or use a new output path.'
}
$controlDirectory = Join-Path $ConfigDirectory ('launch-' + $Mode)
New-Item -ItemType Directory -Path $controlDirectory -Force | Out-Null
if ($Resume -and (Test-Path -LiteralPath (Join-Path $specification.output 'STOP'))) {
    Remove-Item -LiteralPath (Join-Path $specification.output 'STOP')
}
$logPath = Join-Path $controlDirectory ('startup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
$nativeArguments = @('-B','-X','utf8', (Join-Path $PSScriptRoot 'train_nbnm.py'), '--config', $configuration, '--execute', '--log-file', $logPath)
if ($Resume) { $nativeArguments += @('--resume', $Resume) }
$processInfo = [System.Diagnostics.ProcessStartInfo]::new()
$processInfo.FileName = $Python
$processInfo.UseShellExecute = $false
$processInfo.CreateNoWindow = $true
$processInfo.WorkingDirectory = $projectRoot
foreach ($argument in $nativeArguments) { $processInfo.ArgumentList.Add($argument) }
$process = [System.Diagnostics.Process]::Start($processInfo)
@{ PID=$process.Id; Created=$process.StartTime.ToUniversalTime().ToString('o'); Config=$configuration; Output=$specification.output; Log=$logPath } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $controlDirectory 'dispatch.json') -Encoding utf8
Write-Output ('Started NBNM ' + $Mode + ' PID ' + $process.Id)
Write-Output ('Progress: ' + (Join-Path $specification.output 'progress.json'))
Write-Output ('Startup log: ' + $logPath)
