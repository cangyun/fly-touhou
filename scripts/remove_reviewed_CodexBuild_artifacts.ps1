param(
    [Parameter(Mandatory = $true)][string]$PlanPath,
    [Parameter(Mandatory = $true)][string]$ExpectedSha256,
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$env:TEMP = './outputs\tmp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
if ($PSVersionTable.PSVersion.Major -lt 7) { throw 'PowerShell7 required' }

$planFile = (Resolve-Path -LiteralPath $PlanPath).ProviderPath
$actualHash = (Get-FileHash -LiteralPath $planFile -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $ExpectedSha256.ToLowerInvariant()) { throw 'Reviewed plan hash changed' }
$plan = Get-Content -LiteralPath $planFile -Raw -Encoding utf8 | ConvertFrom-Json
if ($plan.schema -ne 'CodexBuild-user-authorized-expired-artifact-cleanup-v1') { throw 'Wrong cleanup schema' }
$authorizedRoot = [IO.Path]::GetFullPath('./outputs/legacy-build').TrimEnd('\')
$root = (Resolve-Path -LiteralPath $plan.root).ProviderPath.TrimEnd('\')
$expectedRoot = [IO.Path]::GetFullPath('./outputs/legacy-build\experiments').TrimEnd('\')
if (-not [string]::Equals($root, $expectedRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Refuse any target root except the explicitly reviewed experiments directory'
}
$boundary = $root + '\'
$checkedParents = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$critical = 'best|final|initial|latest|parent|baseline|selected|failed|proof|qualif|evidence'

function Assert-ParentChain([string]$Directory) {
    $cursor = [IO.Path]::GetFullPath($Directory).TrimEnd('\')
    while ($true) {
        if (-not ($cursor.StartsWith($authorizedRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
            [string]::Equals($cursor, $authorizedRoot, [StringComparison]::OrdinalIgnoreCase))) {
            throw "Directory escaped the authorized parent: $cursor"
        }
        if ($checkedParents.Add($cursor)) {
            $entry = Get-Item -LiteralPath $cursor -Force
            if (-not $entry.PSIsContainer -or ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw "Unexpected directory or reparse point: $cursor"
            }
        }
        if ([string]::Equals($cursor, $authorizedRoot, [StringComparison]::OrdinalIgnoreCase)) { break }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
}

function Assert-ReviewedFile($Record, [bool]$RemovalCandidate) {
    $path = [IO.Path]::GetFullPath([string]$Record.path)
    if (-not $path.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase)) {
        throw "File escaped the reviewed root: $path"
    }
    Assert-ParentChain ([IO.Path]::GetDirectoryName($path))
    $entry = Get-Item -LiteralPath $path -Force
    if ($entry.PSIsContainer -or ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Refuse a directory,link or reparse point: $path"
    }
    if ($entry.Length -ne [long]$Record.bytes -or
        $entry.LastWriteTimeUtc.ToFileTimeUtc() -ne [long]$Record.mtime_filetime) {
        throw "File changed after inventory: $path"
    }
    if ($RemovalCandidate) {
        $parentName = [IO.Path]::GetFileName([IO.Path]::GetDirectoryName($path))
        $name = [IO.Path]::GetFileName($path)
        $isState = $parentName -eq 'rollouts' -and $name -match '^\d{6}\.states\.pt(?:\.gz)?$'
        $isCheckpoint = $parentName -in @('checkpoints', 'epochs') -and
            $name -match '^(?:epoch[_-]\d+.*|checkpoint[_-]\d+.*|\d{6})\.pt$'
        if (-not ($isState -or $isCheckpoint) -or $name -match $critical) {
            throw "Refuse a log,source,trajectory or protected model: $path"
        }
        if ($entry.LastWriteTimeUtc -ge [DateTime]::Parse('2026-10-21T00:00:00+08:00').ToUniversalTime()) {
            throw "Refuse a recent artifact: $path"
        }
    }
}

Assert-ParentChain $root
foreach ($record in $plan.delete) { Assert-ReviewedFile $record $true }
foreach ($record in $plan.preserve) { Assert-ReviewedFile $record $false }
if (-not $Execute) {
    [pscustomobject]@{
        status = 'preflight_passed_no_deletions'
        root = $root
        delete_files = $plan.expected_deleted_files
        retained_files = $plan.preserve.Count
        expected_freed_allocated_GiB = [Math]::Round($plan.expected_deleted_allocated_bytes / 1GB, 3)
    } | ConvertTo-Json -Compress
    exit 0
}

$receiptPath = Join-Path ([IO.Path]::GetDirectoryName($planFile)) 'cleanup_receipt.json'
if (Test-Path -LiteralPath $receiptPath) { throw 'Preserve an existing cleanup receipt;do not execute twice' }
$before = [IO.DriveInfo]::new('J:\').AvailableFreeSpace
$deleted = [Collections.Generic.List[object]]::new()
$failures = [Collections.Generic.List[object]]::new()
foreach ($record in $plan.delete) {
    try {
        Assert-ReviewedFile $record $true
        Remove-Item -LiteralPath ([string]$record.path) -Force -ErrorAction Stop
        $deleted.Add([pscustomobject]@{
            path = $record.path
            bytes = [long]$record.bytes
            allocated_bytes = [long]$record.allocated_bytes
            kind = $record.kind
        })
    } catch {
        $failures.Add([pscustomobject]@{ path = $record.path; error = $_.Exception.Message })
    }
}
$retainedVerified = 0
$retainedFailures = [Collections.Generic.List[object]]::new()
foreach ($record in $plan.preserve) {
    try { Assert-ReviewedFile $record $false; $retainedVerified++ }
    catch { $retainedFailures.Add([pscustomobject]@{ path = $record.path; error = $_.Exception.Message }) }
}
$after = [IO.DriveInfo]::new('J:\').AvailableFreeSpace
$receipt = [ordered]@{
    schema = 'CodexBuild-user-authorized-cleanup-receipt-v1'
    utc = [DateTime]::UtcNow.ToString('o')
    plan = $planFile
    plan_sha256 = $actualHash
    root = $root
    deleted_files = $deleted.Count
    deleted_logical_bytes = [long](($deleted | Measure-Object -Property bytes -Sum).Sum)
    deleted_allocated_bytes = [long](($deleted | Measure-Object -Property allocated_bytes -Sum).Sum)
    disk_free_before = $before
    disk_free_after = $after
    disk_free_delta = $after - $before
    retained_files_verified_unchanged = $retainedVerified
    deletion_errors = @($failures.ToArray())
    retained_file_errors = @($retainedFailures.ToArray())
    deleted = @($deleted.ToArray())
    preserved_sources_tools_and_game_trajectories = $true
    preserved_logs_and_named_critical_models = $true
    current_tests_modified = $false
}
$temporaryReceipt = $receiptPath + '.tmp'
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporaryReceipt -Encoding utf8NoBOM
Move-Item -LiteralPath $temporaryReceipt -Destination $receiptPath
[pscustomobject]@{
    status = 'reviewed_cleanup_finished'
    deleted_files = $deleted.Count
    freed_allocated_GiB = [Math]::Round($receipt.deleted_allocated_bytes / 1GB, 3)
    disk_free_after_GiB = [Math]::Round($after / 1GB, 3)
    retained_files_verified = $retainedVerified
    deletion_errors = $failures.Count
    retained_file_errors = $retainedFailures.Count
    receipt = $receiptPath
} | ConvertTo-Json -Compress
if ($failures.Count -or $retainedFailures.Count) { exit 2 }
