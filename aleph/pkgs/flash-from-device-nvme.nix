# Wrap jetpack-nixos flashFromDevice with NVMe (9:0 / 12:*) write support.
{
  lib,
  python3,
  runCommand,
  flashFromDevice,
}:
runCommand "flash-from-device" {
  meta = {
    mainProgram = "flash-from-device";
    description = "jetpack flashFromDevice with Aleph NVMe support";
  };
  nativeBuildInputs = [python3];
} ''
  mkdir -p $out/bin
  cp ${lib.getExe flashFromDevice} $out/bin/flash-from-device
  chmod +w $out/bin/flash-from-device
  python3 ${./patch-flash-from-device-nvme.py} $out/bin/flash-from-device
  chmod +x $out/bin/flash-from-device
''
