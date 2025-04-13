{pkgs, ...}: let
  pythonPackages = p:
    with p; [
      pipx
      pip
      virtualenv
      numpy
      # opencv4
      # onnxruntime
    ];
in {
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
    nvidia-jetpack.cudaPackages.cudatoolkit
    nvidia-jetpack.cudaPackages.tensorrt
    nvidia-jetpack.l4t-tools
    nvidia-jetpack.l4t-gstreamer
    (python3.withPackages pythonPackages)
    (v4l-utils.override {withGUI = false;})
    (writeShellScriptBin "reset-mcu" (builtins.readFile ../scripts/reset-mcu.sh))
    (writeShellScriptBin "flash-mcu" (builtins.readFile ../scripts/flash-mcu.sh))
  ];
  programs.fish.enable = true;
}
