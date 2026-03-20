{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.aleph.dev;
  pythonPackages = ps:
    (with ps; [
      pipx
      pip
      virtualenv
      numpy
      wheel
      tqdm
      matplotlib
    ])
    ++ builtins.concatLists (map (packageSet: packageSet ps) cfg.extraPythonPackageSets);
in {
  options.aleph.dev.extraPythonPackageSets = lib.mkOption {
    type = lib.types.listOf lib.types.raw;
    default = [];
    internal = true;
    description = "Internal hook for additive Aleph modules to extend the shared Python environment.";
  };

  config = {
    nixpkgs.config.allowUnfree = true;

    environment.variables = {
      CONTAINER_HOST = "unix:///run/podman/podman.sock";
      GST_PLUGIN_PATH = with pkgs;
        map (p: "${lib.getOutput "lib" p}/lib/gstreamer-1.0") [
          gst_all_1.gstreamer
          aravis
          elodinsink
        ];
    };

    virtualisation.podman = {
      enable = true;
      dockerCompat = true;
      dockerSocket.enable = true;
    };

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
      rsync
      gnumake
      pciutils
      usbutils
      nvme-cli
      vim
      htop
      dtc
      btop
      dpkg
      opencv
      lsof
      gst_all_1.gstreamer
      gst_all_1.gst-plugins-base
      gst_all_1.gst-plugins-good
      gst_all_1.gst-plugins-bad
      gst_all_1.gst-plugins-ugly
      gst_all_1.gst-plugins-rs
      aravis
      (python312.withPackages pythonPackages)
      (v4l-utils.override {withGUI = false;})
      # Networking
      tcpdump
      ethtool
      wget
      iperf3
      i2c-tools
      # Utilities for interfacing with the MCU
      (writeShellScriptBin "reset-mcu" (builtins.readFile ../scripts/reset-mcu.sh))
      (writeShellScriptBin "flash-mcu" (builtins.readFile ../scripts/flash-mcu.sh))
      (writeShellScriptBin "aleph-scan" (builtins.readFile ../scripts/aleph-scan.sh))
      aleph-status
      video-streamer
      elodinsink
    ];
  };
}
