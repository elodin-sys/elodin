$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# 5.1 has no RuntimeInformation.OSArchitecture; WOW64 sets PROCESSOR_ARCHITEW6432.
$osArch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
$processArch = $env:PROCESSOR_ARCHITECTURE
Write-Host "OS architecture: $osArch; process architecture: $processArch"
Write-Host "whoami: $(whoami)"
if ($osArch -ne 'AMD64') { throw "Expected Windows 11 x64, got $osArch." }

Get-Command git | Format-List Source
& git --version
if ($LASTEXITCODE -ne 0) { throw 'Git failed.' }
& git lfs version
if ($LASTEXITCODE -ne 0) { throw 'git-lfs failed.' }

$cargo = Get-Command cargo -ErrorAction Stop
Write-Host "cargo: $($cargo.Source)"
& cargo --version
if ($LASTEXITCODE -ne 0) { throw 'cargo failed.' }

$dist = Get-Command dist -ErrorAction Stop
Write-Host "dist: $($dist.Source)"
& dist --version
if ($LASTEXITCODE -ne 0) { throw 'dist failed.' }

$link = Get-Command link.exe -ErrorAction Stop
Write-Host "link.exe: $($link.Source)"
if ($link.Source -notmatch 'Hostx64\\x64\\link\.exe$') {
    throw "link.exe must be the x64-hosted x64 MSVC linker, got $($link.Source)"
}

$candle = Get-Command candle.exe -ErrorAction Stop
Write-Host "candle.exe: $($candle.Source)"
& (Join-Path $PSScriptRoot 'wix-check.ps1')

$PSVersionTable
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber
Write-Host 'Windows x64 runner environment: PASS'
