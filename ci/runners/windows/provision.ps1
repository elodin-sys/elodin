# Requires: elevated PowerShell on Windows 11 x64, as a local administrator.
# Env:
#   RUNNER_TOKEN          required only for registration (mint with ci/runners/bin/mint-token.sh)
#   CI_BUILD_PASSWORD     required (password for the ci-build service account)
#   CI_BUILD_USER         optional, default ci-build
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Keep in sync with ci/runners/versions.env
$RepoUrl = 'https://github.com/elodin-sys/elodin'
$RunnerVersion = '2.337.0'
$RunnerSha256 = '1150692afa94e71f872017e254ea55b6eece1eece3fe7e3a6d4c93d0a1b85cfc'
$DistVersion = '0.28.0'
$RustToolchain = '1.98.0'

if (-not [Environment]::Is64BitOperatingSystem) { throw 'Expected 64-bit Windows.' }
# 5.1 has no RuntimeInformation.OSArchitecture; WOW64 sets PROCESSOR_ARCHITEW6432.
$osArch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($osArch -ne 'AMD64') { throw "Expected Windows 11 x64, got $osArch." }

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]$identity
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run provision.ps1 from an elevated PowerShell session.'
}

if (-not $env:CI_BUILD_PASSWORD) { throw 'Set CI_BUILD_PASSWORD for the ci-build service account.' }
$buildUser = if ($env:CI_BUILD_USER) { $env:CI_BUILD_USER } else { 'ci-build' }

function Assert-LastExit {
    param([string]$Name)
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit $LASTEXITCODE" }
}

Set-ExecutionPolicy -Scope LocalMachine Bypass -Force
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1 -Type DWord
powercfg /change standby-timeout-ac 0
Assert-LastExit 'powercfg standby-timeout-ac'
powercfg /change hibernate-timeout-ac 0
Assert-LastExit 'powercfg hibernate-timeout-ac'

$runnerRoot = 'C:\actions-runner'
New-Item -ItemType Directory -Force -Path $runnerRoot | Out-Null
try {
    Add-MpPreference -ExclusionPath $runnerRoot
} catch {
    Write-Warning "Could not add Defender exclusion for ${runnerRoot}: $_"
}

if (-not (Get-LocalUser -Name $buildUser -ErrorAction SilentlyContinue)) {
    Write-Host "Creating local user $buildUser..."
    $securePassword = ConvertTo-SecureString $env:CI_BUILD_PASSWORD -AsPlainText -Force
    New-LocalUser -Name $buildUser `
        -Password $securePassword `
        -FullName $buildUser `
        -Description 'GitHub Actions runner service account' `
        -PasswordNeverExpires `
        -UserMayNotChangePassword `
        -AccountNeverExpires | Out-Null
}

$specialAccounts = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon\SpecialAccounts\UserList'
if (-not (Test-Path $specialAccounts)) {
    New-Item -Path $specialAccounts -Force | Out-Null
}
New-ItemProperty -Path $specialAccounts -Name $buildUser -Value 0 -PropertyType DWord -Force | Out-Null

if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host 'Installing Chocolatey...'
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}

choco install git.install --yes --params '/GitAndUnixToolsOnPath /WindowsTerminal'
Assert-LastExit 'choco install git'
choco install git-lfs --yes
Assert-LastExit 'choco install git-lfs'
choco install cmake --yes
Assert-LastExit 'choco install cmake'
choco install protoc --yes
Assert-LastExit 'choco install protoc'
choco install wixtoolset --yes
Assert-LastExit 'choco install wixtoolset'

$vsParams = @(
    '--add Microsoft.VisualStudio.Workload.VCTools'
    '--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
    '--includeRecommended'
) -join ' '
choco install visualstudio2022buildtools --yes --package-parameters $vsParams
Assert-LastExit 'choco install visualstudio2022buildtools'

$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
git lfs install
Assert-LastExit 'git lfs install'

# Shared toolchain so the ci-build service account can see rustc/dist.
$toolRoot = Join-Path $runnerRoot 'tools'
$cargoHome = Join-Path $toolRoot 'cargo'
$rustupHome = Join-Path $toolRoot 'rustup'
New-Item -ItemType Directory -Force -Path $cargoHome, $rustupHome | Out-Null
$env:CARGO_HOME = $cargoHome
$env:RUSTUP_HOME = $rustupHome
$cargoBin = Join-Path $cargoHome 'bin'

$rustup = Join-Path $env:TEMP 'rustup-init.exe'
Invoke-WebRequest -Uri 'https://win.rustup.rs/x86_64' -OutFile $rustup
& $rustup -y --default-toolchain $RustToolchain --default-host x86_64-pc-windows-msvc
Assert-LastExit 'rustup-init'
$env:Path = "$cargoBin;$env:Path"
rustup default $RustToolchain
Assert-LastExit 'rustup default'

if (-not (Get-Command dist -ErrorAction SilentlyContinue)) {
    Write-Host "Installing cargo-dist $DistVersion..."
    Invoke-Expression (Invoke-RestMethod "https://github.com/axodotdev/cargo-dist/releases/download/v$DistVersion/cargo-dist-installer.ps1")
    $env:Path = "$cargoBin;$env:Path"
}
if (-not (Get-Command dist -ErrorAction SilentlyContinue)) {
    throw "dist not on PATH after cargo-dist $DistVersion installer."
}

icacls $runnerRoot /grant "${buildUser}:(OI)(CI)M" /T | Out-Null

$wixCandidates = @(
    'C:\Program Files (x86)\WiX Toolset v3.14\bin'
    'C:\Program Files (x86)\WiX Toolset v3.11\bin'
    'C:\Program Files\WiX Toolset v3.14\bin'
)
$wixBin = $wixCandidates | Where-Object { Test-Path (Join-Path $_ 'candle.exe') } | Select-Object -First 1
if (-not $wixBin) { throw 'WiX candle.exe not found after wixtoolset install.' }
[Environment]::SetEnvironmentVariable('WIX', (Split-Path $wixBin -Parent), 'Machine')

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "vswhere missing at $vswhere" }
$vsRoot = & $vswhere -latest -products * -property installationPath
if (-not $vsRoot) { throw 'Visual Studio Build Tools installation path not found.' }
$link = Get-ChildItem -Path (Join-Path $vsRoot 'VC\Tools\MSVC') -Recurse -Filter link.exe |
    Where-Object { $_.FullName -match 'Hostx64\\x64\\link\.exe$' } |
    Select-Object -First 1
if (-not $link) { throw 'MSVC Hostx64\x64\link.exe not found. Confirm the x64 toolset is installed.' }
$msvcBin = $link.Directory.FullName

$machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
foreach ($dir in @($msvcBin, $wixBin, $cargoBin, 'C:\Program Files\Git\cmd')) {
    if ($machinePath -notlike "*$dir*") {
        $machinePath = "$dir;$machinePath"
    }
}
[Environment]::SetEnvironmentVariable('Path', $machinePath, 'Machine')
$env:Path = $machinePath

Set-Location $runnerRoot
if (-not (Test-Path (Join-Path $runnerRoot 'config.cmd'))) {
    $zip = "actions-runner-win-x64-$RunnerVersion.zip"
    $url = "https://github.com/actions/runner/releases/download/v$RunnerVersion/$zip"
    Invoke-WebRequest -Uri $url -OutFile $zip
    $actual = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
    if ($actual -ne $RunnerSha256) { throw "Runner SHA-256 mismatch: $actual != $RunnerSha256" }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory((Join-Path $runnerRoot $zip), $runnerRoot)
    Remove-Item $zip
}

@(
    $msvcBin
    $wixBin
    $cargoBin
    'C:\Program Files\CMake\bin'
    'C:\Program Files\Git\cmd'
) | Set-Content -Path (Join-Path $runnerRoot '.path')
@(
    "CARGO_HOME=$cargoHome"
    "RUSTUP_HOME=$rustupHome"
) | Set-Content -Path (Join-Path $runnerRoot '.env')

if (-not (Test-Path (Join-Path $runnerRoot '.runner'))) {
    # Token lives one hour; prompt here so it is minted after the long VS install.
    $runnerToken = if ($env:RUNNER_TOKEN) { $env:RUNNER_TOKEN.Trim() } else { '' }
    if (-not $runnerToken) {
        Write-Host 'Mint a registration token (valid 1 hour) on an admin machine:'
        Write-Host '  RUNNER_TOKEN=$(ci/runners/bin/mint-token.sh)'
        $runnerToken = (Read-Host 'Paste RUNNER_TOKEN').Trim()
    }
    if (-not $runnerToken) { throw 'Set RUNNER_TOKEN (ci/runners/bin/mint-token.sh) and re-run.' }
    & .\config.cmd --unattended `
        --url $RepoUrl `
        --token $runnerToken `
        --name ci-windows-x64 `
        --labels 'ci,ci-windows-x64' `
        --work '_work' `
        --runasservice `
        --windowslogonaccount ".\$buildUser" `
        --windowslogonpassword $env:CI_BUILD_PASSWORD
    Assert-LastExit 'config.cmd'
}

& (Join-Path $PSScriptRoot 'seed-python.ps1')

Get-Service 'actions.runner.*' | Format-Table Name, Status
Write-Host 'Windows runner registered. Confirm it is online in GitHub Settings > Actions > Runners.'
