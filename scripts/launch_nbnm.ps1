param(
    [ValidateSet('route','baseline')][string]$Mode = 'route',
    [string]$ConfigDirectory = '',
    [string]$Resume = '',
    [ValidateRange(0,10000)][int]$AdditionalSegments = 0,
    [switch]$PreserveBudget,
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
if ($PreserveBudget -and $AdditionalSegments -gt 0) {
    throw 'PreserveBudget cannot be combined with additional training segments.'
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$localPath = Join-Path $projectRoot 'outputs/nbnm-local.json'
if (Test-Path -LiteralPath $localPath) {
    $localSettings = Get-Content -LiteralPath $localPath -Raw | ConvertFrom-Json
    if (-not $ConfigDirectory) { $ConfigDirectory = $localSettings.config_directory }
    if (-not $Python) { $Python = $localSettings.python }
}
if (-not $ConfigDirectory) { $ConfigDirectory = Join-Path $projectRoot 'outputs/nbnm-preparation' }
if (-not $Python) { $Python = $env:FLY_TOUHOU_PYTHON }
if (-not $Python) { $Python = Join-Path $projectRoot '.venv/Scripts/python.exe' }
$Python = (Resolve-Path -LiteralPath $Python).Path
$configuration = (Resolve-Path -LiteralPath (Join-Path $ConfigDirectory ($Mode + '.json'))).Path
if ($Resume) { $Resume = (Resolve-Path -LiteralPath $Resume).Path }
$specification = Get-Content -LiteralPath $configuration -Raw | ConvertFrom-Json
$progressPath = Join-Path $specification.output 'progress.json'
if (Test-Path -LiteralPath $progressPath) {
    $share = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $stream = [IO.File]::Open($progressPath,[IO.FileMode]::Open,[IO.FileAccess]::Read,$share)
    $reader = [IO.StreamReader]::new($stream)
    try { $lastProgress = $reader.ReadToEnd() | ConvertFrom-Json }
    finally { $reader.Dispose() }
    $running = Get-Process -Id $lastProgress.PID -ErrorAction SilentlyContinue
    if ($running -and [math]::Abs(($running.StartTime.ToUniversalTime() - [datetime]'1970-01-01Z').TotalSeconds - $lastProgress.create_time) -lt 2) {
        throw 'This NBNM trainer is already running. Use its existing status panel.'
    }
}
if (-not $Resume) {
    $latest = Join-Path $specification.output 'latest-training.pt'
    if (Test-Path -LiteralPath $latest) { $Resume = (Resolve-Path -LiteralPath $latest).Path }
    elseif ($specification.resume_checkpoint) {
        $Resume = (Resolve-Path -LiteralPath $specification.resume_checkpoint).Path
    }
}
if ($Resume -and -not $PreserveBudget -and $AdditionalSegments -eq 0 -and $lastProgress -and
    $lastProgress.completed_decisions -ge $specification.decision_budget) {
    $AdditionalSegments = if ($specification.resume_extension_segments) { [int]$specification.resume_extension_segments } else { 8 }
}
if ($AdditionalSegments -gt 0) {
    $specification.decision_budget += $AdditionalSegments * $specification.segment_decisions
    $specification | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $configuration -Encoding utf8
}
$validationArguments = @('-B','-X','utf8', (Join-Path $PSScriptRoot 'train_nbnm.py'), '--config', $configuration)
if ($Resume) { $validationArguments += @('--resume', $Resume) }
& $Python @validationArguments
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
