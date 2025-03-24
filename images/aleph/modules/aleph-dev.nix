{pkgs, ...}: {
  environment.systemPackages = with pkgs; [
    libgpiod_1
    dfu-util
    gcc-arm-embedded
    stm32flash
    tio
    fish
    neovim
    git
    uv
    python312
    ripgrep
    tcpdump
    iperf3
    v4l-utils
    (writeShellScriptBin "reset-mcu" (builtins.readFile ../scripts/reset-mcu.sh))
    (writeShellScriptBin "flash-mcu" (builtins.readFile ../scripts/flash-mcu.sh))
  ];
}
