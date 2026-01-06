{pkgs, ...}: let
  onnxruntime-gpu-wheel = ps:
    ps.buildPythonPackage {
      pname = "onnxruntime-gpu";
      version = "1.23.0";
      format = "wheel";

      src = pkgs.fetchurl {
        url = "https://pypi.jetson-ai-lab.io/jp6/cu129/+f/2e3/a07114007df15/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl";
        sha256 = "2e3a07114007df15db673852d798d6f47f91362f0ac084d6fa04e414a06dc25e";
      };

      propagatedBuildInputs = with ps; [
        coloredlogs
        flatbuffers
        numpy
        packaging
        protobuf
        sympy
      ];
    };
  cudainfo = pkgs.writeScriptBin "cudainfo" (builtins.readFile ../scripts/cudainfo.py);
  pythonPackages = ps:
    with ps; [
      pipx
      pip
      virtualenv
      numpy
      wheel
      onnx
      tqdm
      matplotlib
      pycuda
      (onnxruntime-gpu-wheel ps)
    ];
in {
  nixpkgs.config = {
    allowUnfree = true;
    cudaSupport = true;
    cudaCapabilities = ["7.2" "8.7"];
  };

  environment.variables = with pkgs; {
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      stdenv.cc.cc.lib
      cudaPackages.cudatoolkit
      cudaPackages.cudnn
      gst_all_1.gstreamer
      nvidia-jetpack.l4t-cuda
      nvidia-jetpack.l4t-gstreamer
      nvidia-jetpack.l4t-multimedia
      nvidia-jetpack.l4t-camera
    ];
    NVCC_PREPEND_FLAGS = "--compiler-bindir ${pkgs.gcc11}/bin/gcc";
    CONTAINER_HOST = "unix:///run/podman/podman.sock";
    GST_PLUGIN_PATH = lib.makeSearchPathOutput "lib" "lib/gstreamer-1.0" [
      gst_all_1.gstreamer
      aravis
      deepstream
    ];
  };
  hardware.graphics.enable = true;
  virtualisation.podman = {
    enable = true;
    # TODO: replace with `hardware.nvidia-container-toolkit.enable` when it works (https://github.com/nixos/nixpkgs/issues/344729).
    enableNvidia = true;
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
    cudainfo
    lsof
    gst_all_1.gstreamer
    gst_all_1.gst-plugins-base
    gst_all_1.gst-plugins-good
    gst_all_1.gst-plugins-bad
    gst_all_1.gst-plugins-ugly
    gst_all_1.gst-plugins-rs
    aravis
    nvidia-jetpack.l4t-gstreamer
    nvidia-jetpack.l4t-multimedia
    cudaPackages.cuda_nvcc
    nvidia-jetpack.samples.cuda-test
    nvidia-jetpack.samples.cudnn-test
    (python311.withPackages pythonPackages)
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
    deepstream
  ];
}
