param(
    [Parameter(Mandatory = $true)][string]$Plan
)

$ErrorActionPreference = 'Stop'
$planData = Get-Content -LiteralPath $Plan -Raw -Encoding UTF8 | ConvertFrom-Json
$sourceRoot = (Resolve-Path -LiteralPath $planData.sourceRoot).Path
$destinationRoot = (Resolve-Path -LiteralPath $planData.destinationRoot).Path
if (-not $sourceRoot.Equals($planData.sourceRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected source root' }
if (-not $destinationRoot.Equals($planData.destinationRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected destination root' }
$state = @{}
foreach ($job in $planData.jobs) { $state[$job.name] = 'waiting' }

function Save-State {
    $record = @{ updatedUtc = [datetime]::UtcNow.ToString('o'); jobs = $state }
    $temp = $planData.statusFile + '.tmp'
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temp -Encoding UTF8
    Move-Item -LiteralPath $temp -Destination $planData.statusFile -Force
}

function Test-JobRunning($job) {
    foreach ($identity in $job.waitFor) {
        $proc = Get-Process -Id $identity.pid -ErrorAction SilentlyContinue
        if ($proc -and $proc.StartTime.ToUniversalTime().Ticks -eq [long]$identity.startTicks) { return $true }
    }
    # A wrapper can exit unexpectedly while its owned evaluator is still live.
    foreach ($relative in @('parent\progress.json', 'candidate\progress.json')) {
        $progress = Join-Path (Join-Path $sourceRoot $job.name) $relative
        if (Test-Path -LiteralPath $progress -PathType Leaf) {
            try {
                $observed = Get-Content -LiteralPath $progress -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($observed.PID -and (Get-Process -Id $observed.PID -ErrorAction SilentlyContinue)) { return $true }
            } catch {}
        }
    }
    return $false
}

function Move-FinishedJob($job) {
    $source = (Resolve-Path -LiteralPath (Join-Path $sourceRoot $job.name)).Path
    if ((Split-Path -Parent $source) -ne $sourceRoot) { throw 'Source escapes the intended artifact directory' }
    if ((Get-Item -LiteralPath $source -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        return 'already-linked'
    }
    $destination = Join-Path $destinationRoot $job.name
    $hold = $source + '.migration-hold'
    if ((Test-Path -LiteralPath $destination) -or (Test-Path -LiteralPath $hold)) { throw 'Destination or hold already exists' }
    $files = @(Get-ChildItem -LiteralPath $source -Recurse -File -Force)
    if (@(Get-ChildItem -LiteralPath $source -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Nested link in source' }
    Copy-Item -LiteralPath $source -Destination $destination -Recurse -ErrorAction Stop
    if (@(Get-ChildItem -LiteralPath $destination -Recurse -File -Force).Count -ne $files.Count) { throw 'Copied file count differs' }
    foreach ($file in $files) {
        $relative = $file.FullName.Substring($source.Length + 1)
        $copy = Join-Path $destination $relative
        if (-not (Test-Path -LiteralPath $copy -PathType Leaf) -or (Get-Item -LiteralPath $copy).Length -ne $file.Length) { throw ('Copy mismatch: ' + $relative) }
    }
    Rename-Item -LiteralPath $source -NewName ($job.name + '.migration-hold')
    try {
        New-Item -ItemType Junction -Path $source -Target $destination -ErrorAction Stop | Out-Null
        if (-not (Test-Path -LiteralPath (Join-Path $source $job.anchor) -PathType Leaf)) { throw 'Original access path did not resolve' }
    } catch {
        if (Test-Path -LiteralPath $source) { [IO.Directory]::Delete($source) }
        Rename-Item -LiteralPath $hold -NewName $job.name
        throw
    }
    $resolvedHold = (Resolve-Path -LiteralPath $hold).Path
    if (-not $resolvedHold.Equals($hold, [StringComparison]::OrdinalIgnoreCase) -or -not $resolvedHold.StartsWith($sourceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe migration hold' }
    Remove-Item -LiteralPath $resolvedHold -Recurse -Force
    return 'migrated'
}

Save-State
while (@($state.Values | Where-Object { $_ -eq 'waiting' }).Count -gt 0) {
    foreach ($job in $planData.jobs) {
        if ($state[$job.name] -ne 'waiting' -or (Test-JobRunning $job)) { continue }
        try { $state[$job.name] = Move-FinishedJob $job }
        catch { $state[$job.name] = 'failed: ' + $_.Exception.Message }
        Save-State
    }
    if (@($state.Values | Where-Object { $_ -eq 'waiting' }).Count -gt 0) { Start-Sleep -Seconds 20 }
}
