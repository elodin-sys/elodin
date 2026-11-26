{pkgs, ...}:
pkgs.writeShellApplication {
  name = "flash-mcu";
  runtimeInputs = [
    pkgs.coreutils
    pkgs.gcc-arm-embedded
    pkgs.lsof
    pkgs.openocd
    pkgs.procps
    pkgs.python3
    pkgs.stm32flash
    pkgs.systemd
    pkgs.reset-mcu
  ];
  text = builtins.readFile ../scripts/flash-mcu.sh;
}
