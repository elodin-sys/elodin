{pkgs, ...}: let
  onnxruntime-gpu-wheel = ps:
    ps.buildPythonPackage {
      pname = "onnxruntime-gpu";
      version = "1.16.3";
      format = "wheel";

      src = pkgs.fetchurl {
        url = "https://pypi.jetson-ai-lab.dev/jp5/cu114/+f/43e/f0cec5f026159/onnxruntime_gpu-1.16.3-cp311-cp311-linux_aarch64.whl";
        sha256 = "43ef0cec5f026159306e69540138f457ecbc8eb0282d1f7166761e7fbc84288e";
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
      cudaPackages.tensorrt
      cudaPackages.vpi2
      gst_all_1.gstreamer
      nvidia-jetpack.l4t-cuda
      nvidia-jetpack.l4t-gstreamer
      nvidia-jetpack.l4t-multimedia
      nvidia-jetpack.l4t-camera
    ];
    NVCC_PREPEND_FLAGS = "--compiler-bindir ${pkgs.gcc11}/bin/gcc";
    NVCC_APPEND_FLAGS = "-I${pkgs.cudaPackages.cuda_cudart.include}/include";
    CONTAINER_HOST = "unix:///run/podman/podman.sock";
    GST_PLUGIN_PATH = lib.makeSearchPathOutput "lib" "lib/gstreamer-1.0" [
      gst_all_1.gstreamer
      aravis
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
    # Utilities for interfacing with the MCU
    (writeShellScriptBin "reset-mcu" (builtins.readFile ../scripts/reset-mcu.sh))
    (writeShellScriptBin "flash-mcu" (builtins.readFile ../scripts/flash-mcu.sh))
    (writeShellScriptBin "aleph-scan" (builtins.readFile ../scripts/aleph-scan.sh))
    aleph-status
    video-streamer
    elodinsink
  ];
}
