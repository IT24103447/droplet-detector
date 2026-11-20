param(
    [Parameter(Mandatory)][string]$Plan,
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
$sourcePath = 'D:\TransitionX\test_records\test_records\test_records'
$source = (Get-Item -LiteralPath $sourcePath).FullName
if ($source -ne $sourcePath) { throw 'Unexpected source path.' }
$allowed = @('Visible leaks', 'No visible leaks in sampled frames', 'Needs review', 'Visible', 'No Visible')
$entries = @(Get-Content -LiteralPath $Plan -Raw | ConvertFrom-Json)
$names = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$destinations = @{}
foreach ($bucket in $allowed) {
    $destination = Join-Path $source $bucket
    if (Test-Path -LiteralPath $destination) {
        $directory = Get-Item -LiteralPath $destination
        if (-not $directory.PSIsContainer -or ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Unsafe destination: $destination"
        }
    }
    $destinations[$bucket] = $destination
}
foreach ($entry in $entries) {
    $name = [string]$entry.file
    if ($name -ne [IO.Path]::GetFileName($name) -or [IO.Path]::GetExtension($name) -ne '.mp4' -or
        -not $names.Add($name) -or $entry.bucket -notin $allowed) {
        throw "Invalid or duplicate plan entry: $name"
    }
    if ($entry.previous_bucket -and $entry.previous_bucket -notin $allowed) {
        throw "Invalid previous folder for $name."
    }
    $originDirectory = if ($entry.previous_bucket) { $destinations[$entry.previous_bucket] } else { $source }
    $origin = Join-Path $originDirectory $name
    $target = Join-Path $destinations[$entry.bucket] $name
    $hasOrigin = Test-Path -LiteralPath $origin -PathType Leaf
    $hasTarget = Test-Path -LiteralPath $target -PathType Leaf
    if ($hasOrigin -eq $hasTarget) { throw "Ambiguous location for $name." }
    $current = Get-Item -LiteralPath $(if ($hasOrigin) { $origin } else { $target })
    if (($current.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $current.Length -ne $entry.bytes) {
        throw "Source does not match the inventory: $name"
    }
}
$entries | Group-Object bucket | Select-Object Name,Count
if (-not $Apply) { return }
foreach ($bucket in ($entries.bucket | Sort-Object -Unique)) {
    $destination = $destinations[$bucket]
    if (-not (Test-Path -LiteralPath $destination)) {
        New-Item -ItemType Directory -Path $destination | Out-Null
    }
}
$blocked = @()
foreach ($entry in $entries) {
    $originDirectory = if ($entry.previous_bucket) { $destinations[$entry.previous_bucket] } else { $source }
    $origin = Join-Path $originDirectory $entry.file
    if (Test-Path -LiteralPath $origin -PathType Leaf) {
        try {
            Move-Item -LiteralPath $origin -Destination (Join-Path $destinations[$entry.bucket] $entry.file)
        } catch {
            $blocked += $entry.file
            Write-Warning "$($entry.file): $($_.Exception.Message)"
        }
    }
}
foreach ($entry in $entries) {
    if ($entry.file -in $blocked) { continue }
    $target = Join-Path $destinations[$entry.bucket] $entry.file
    $originDirectory = if ($entry.previous_bucket) { $destinations[$entry.previous_bucket] } else { $source }
    if ((Get-Item -LiteralPath $target).Length -ne $entry.bytes -or
        (Test-Path -LiteralPath (Join-Path $originDirectory $entry.file))) {
        throw "Move verification failed: $($entry.file)"
    }
}
Write-Output "Verified $($entries.Count - $blocked.Count) original MP4s at their sorted destinations."
if ($blocked.Count) { exit 1 }
