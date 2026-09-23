# Pre-seeds actions/setup-python's tool cache so jobs never run the
# python-versions installer (InstallAllUsers=1, ~1 min, needs network).
# Run from an elevated PowerShell. Re-run whenever $PythonSeries changes
# (python-version in release.yml).
# Env:
#   CI_BUILD_USER         optional, default ci-build
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Keep in sync with ci/runners/versions.env
$PythonSeries = '3.13'

$runnerRoot = 'C:\actions-runner'
$toolcache = Join-Path $runnerRoot '_work\_tool'
$manifestUrl = 'https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json'
$buildUser = if ($env:CI_BUILD_USER) { $env:CI_BUILD_USER } else { 'ci-build' }

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]$identity
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run seed-python.ps1 from an elevated PowerShell session.'
}
if (-not (Get-LocalUser -Name $buildUser -ErrorAction SilentlyContinue)) {
    throw "Local user $buildUser does not exist; run provision.ps1 first."
}

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$manifest = Invoke-RestMethod -Uri $manifestUrl
$release = $manifest | Where-Object { $_.stable -and $_.version -like "$PythonSeries.*" } | Select-Object -First 1
if (-not $release) { throw "No stable Python $PythonSeries release in the python-versions manifest." }
$file = $release.files | Where-Object { $_.platform -eq 'win32' -and $_.arch -eq 'x64' } | Select-Object -First 1
if (-not $file) { throw "No win32/x64 build for Python $($release.version)." }

$complete = Join-Path $toolcache "Python\$($release.version)\x64.complete"
if (Test-Path $complete) {
    Write-Host "Python $($release.version) already seeded in $toolcache."
} else {
    Write-Host "Seeding Python $($release.version) into $toolcache..."
    New-Item -ItemType Directory -Force -Path $toolcache | Out-Null
    $tmp = Join-Path $env:TEMP "python-versions-$($release.version)"
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    New-Item -ItemType Directory -Path $tmp | Out-Null
    $zip = Join-Path $tmp 'python.zip'
    Invoke-WebRequest -Uri $file.download_url -OutFile $zip
    Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force

    $env:AGENT_TOOLSDIRECTORY = $toolcache
    Push-Location $tmp
    try {
        & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File .\setup.ps1
        if ($LASTEXITCODE -ne 0) { throw "python-versions setup.ps1 failed with exit $LASTEXITCODE" }
    } finally {
        Pop-Location
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

icacls $toolcache /grant "${buildUser}:(OI)(CI)M" /T | Out-Null
Write-Host "setup-python cache ready: $toolcache\Python\$($release.version)\x64"
