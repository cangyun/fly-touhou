param([Parameter(Mandatory=$true)][string]$Inventory,[Parameter(Mandatory=$true)][string]$Receipt)
$ErrorActionPreference='Stop'
$record=Get-Content -LiteralPath $Inventory -Raw | ConvertFrom-Json
$taskTempRoot=[IO.Path]::GetFullPath([string]$record.source_root).TrimEnd('\','/')
$expectedTemp=[IO.Path]::GetFullPath($env:TEMP).TrimEnd('\','/')
if ($taskTempRoot -ne $expectedTemp) { throw 'Unexpected source temp root' }
$taskArchiveRoot=[IO.Path]::GetFullPath([string]$record.archive_root).TrimEnd('\','/')
if ($taskArchiveRoot -ne './outputs\retired-native-temp-20261024') { throw 'Unexpected archive target' }
[void][IO.Directory]::CreateDirectory($taskArchiveRoot)
$rows=@();$movedBytes=0L;$before=(Get-PSDrive -Name C).Free
foreach ($item in $record.items) {
    $sourcePath=[IO.Path]::GetFullPath([string]$item.path)
    if ([IO.Path]::GetDirectoryName($sourcePath) -ne $taskTempRoot -or [IO.Path]::GetFileName($sourcePath) -notmatch '^frida-[0-9a-f]{32}$') { throw 'Temp target outside verified scope' }
    $targetPath=[IO.Path]::GetFullPath((Join-Path $taskArchiveRoot ([IO.Path]::GetFileName($sourcePath))))
    if ([IO.Path]::GetDirectoryName($targetPath) -ne $taskArchiveRoot) { throw 'Archive target escaped scope' }
    if (!(Test-Path -LiteralPath $sourcePath)) { continue }
    if (Test-Path -LiteralPath $targetPath) { throw 'Archive already exists; preserve it' }
    try {
        $sourceItem=Get-Item -LiteralPath $sourcePath -Force
        if ($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refuse reparse point' }
        $entries=@(Get-ChildItem -LiteralPath $sourcePath -Recurse -Force)
        foreach ($entry in $entries) {
            if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refuse nested reparse point' }
            if (!$entry.PSIsContainer) {
                if ($entry.Name -notmatch '^(frida-helper-(arm64|x86|x86_64)\.exe|frida-agent\.dll|dbghelp\.dll|symsrv\.dll)$') { throw 'Unexpected file in disposable cache' }
                $probe=[IO.File]::Open($entry.FullName,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::None)
                $probe.Dispose()
            }
        }
        Move-Item -LiteralPath $sourcePath -Destination $targetPath -ErrorAction Stop
        $movedBytes += [long]$item.bytes
        $rows += [pscustomobject]@{source=$sourcePath;target=$targetPath;bytes=[long]$item.bytes;status='archived'}
    } catch {
        $rows += [pscustomobject]@{source=$sourcePath;status='preserved_or_partial';error=$_.Exception.Message}
    }
}
$result=[pscustomobject]@{archivedBytes=$movedBytes;C_free_before=$before;C_free_after=(Get-PSDrive -Name C).Free;rows=$rows;project_results_or_logs_modified=$false}
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Receipt -Encoding utf8
[pscustomobject]@{archivedBytes=$movedBytes;folders=@($rows | Where-Object status -eq 'archived').Count;C_free_after=(Get-PSDrive -Name C).Free;unmoved=@($rows | Where-Object status -ne 'archived').Count} | ConvertTo-Json -Compress
