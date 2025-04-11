{pkgs, ...}: {
  environment.systemPackages = with pkgs; [
    libgpiod_1
    dfu-util
    gcc
    pkg-config
    gcc-arm-embedded
    stm32flash
    tio
    neovim
    git
    uv
    ripgrep
    tcpdump
    ethtool
    iperf3
    rsync
    gnumake
    pciutils
    usbutils
    nvme-cli
    vim
    htop
    dtc
    (python3.withPackages (python-pkgs:
      with python-pkgs; [
        pipx
        pip
        virtualenv
      ]))
    (v4l-utils.override {withGUI = false;})
    (writeShellScriptBin "reset-mcu" (builtins.readFile ../scripts/reset-mcu.sh))
    (writeShellScriptBin "flash-mcu" (builtins.readFile ../scripts/flash-mcu.sh))
  ];
  programs.fish.enable = true;
}
