{pkgs, ...}:
pkgs.writeShellApplication {
  name = "reset-mcu";
  runtimeInputs = [
    pkgs.coreutils
    pkgs.i2c-tools
    pkgs.libgpiod_1
    pkgs.procps
  ];
  text = builtins.readFile ../scripts/reset-mcu.sh;
}
