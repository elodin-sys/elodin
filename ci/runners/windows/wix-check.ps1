# Builds a one-file MSI with WiX 3 candle + light, ICE validation on, exactly
# as `dist build` does. light.exe's ICE step needs the Windows Installer
# service; it fails with LGHT0216/0217 in some service/user contexts. Run this
# from the runner (smoke.ps1) and interactively (`runas /user:ci-build`) to
# tell those cases apart. Prints the full light output that dist swallows.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$candle = (Get-Command candle.exe -ErrorAction Stop).Source
$light = Join-Path (Split-Path $candle -Parent) 'light.exe'
if (-not (Test-Path $light)) { throw "light.exe not found beside $candle" }
Write-Host "candle: $candle"
Write-Host "light:  $light"
Write-Host "whoami: $(whoami); session: $((Get-Process -Id $PID).SessionId)"

$work = Join-Path ([IO.Path]::GetTempPath()) "wix-check-$PID"
New-Item -ItemType Directory -Path $work -Force | Out-Null
try {
    $payload = Join-Path $work 'payload.txt'
    Set-Content -Path $payload -Value 'wix-check'
    $wxs = Join-Path $work 'check.wxs'
    @"
<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
  <Product Id="*" Name="WixCheck" Language="1033" Version="1.0.0.0"
           Manufacturer="Elodin CI" UpgradeCode="0F2A7C6E-1B4D-4E3A-9C1B-2D3E4F5A6B7C">
    <Package InstallerVersion="500" Compressed="yes" InstallScope="perMachine" />
    <MediaTemplate EmbedCab="yes" />
    <Directory Id="TARGETDIR" Name="SourceDir">
      <Directory Id="ProgramFiles64Folder">
        <Directory Id="INSTALLDIR" Name="WixCheck">
          <Component Id="Main" Guid="7D8E9F0A-1B2C-4D3E-8F4A-5B6C7D8E9F0A" Win64="yes">
            <File Id="Payload" Source="$payload" KeyPath="yes" />
          </Component>
        </Directory>
      </Directory>
    </Directory>
    <Feature Id="Main" Level="1"><ComponentRef Id="Main" /></Feature>
  </Product>
</Wix>
"@ | Set-Content -Path $wxs -Encoding UTF8

    & $candle -nologo -arch x64 -out (Join-Path $work 'check.wixobj') $wxs
    if ($LASTEXITCODE -ne 0) { throw "candle failed ($LASTEXITCODE)" }
    & $light -nologo -out (Join-Path $work 'check.msi') (Join-Path $work 'check.wixobj')
    if ($LASTEXITCODE -ne 0) {
        throw "light failed ($LASTEXITCODE). LGHT0216/0217 => ICE validation cannot reach the Windows Installer service in this context."
    }
    Write-Host 'WiX candle+light with ICE validation: PASS'
} finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
