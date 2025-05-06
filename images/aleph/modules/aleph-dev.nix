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

  tensorflow-wheel = ps:
    ps.buildPythonPackage {
      pname = "tensorflow";
      version = "2.12.0";
      format = "wheel";

      src = pkgs.fetchurl {
        url = "https://developer.download.nvidia.com/compute/redist/jp/v61/tensorflow/tensorflow-2.16.1+nv24.08-cp310-cp310-linux_aarch64.whl";
        sha256 = "sha256-7jzTPCT3XsnUKVgEma6ce/3LSoM7TpLos7EJJln7KHw=";
        # url = "https://developer.download.nvidia.com/compute/redist/jp/v512/tensorflow/tensorflow-2.12.0+nv23.06-cp38-cp38-linux_aarch64.whl";
        # sha256 = "sha256-7QyyCy1hAOX1CFedZwoePDYvICBXPy2I0euxX5+dIMI=";
      };

      propagatedBuildInputs = with ps; [
        absl-py
        astunparse
        flatbuffers
        gast
        google-pasta
        h5py
        libclang
        opt-einsum
        packaging
        protobuf
        setuptools
        six
        termcolor
        typing-extensions
        wrapt
        grpcio
        tensorflow-estimator
        numpy
        ml-dtypes
        requests
        testresources
        future
        mock
        keras-preprocessing
        keras-applications
        pybind11
        cython
      ];
    };

  opencvOverlay = final: prev: {
    opencv4 = prev.opencv4.overrideAttrs (old: {
      cmakeFlags   = (old.cmakeFlags or []) ++ [
        "-DWITH_CUDA=ON"
        "-DWITH_CUDNN=ON"
        "-DCUDA_ARCH_BIN=8.7"
        "-DCUDA_ARCH_PTX=8.7"
      ];
    });
  };

  cudainfo = pkgs.writeScriptBin "cudainfo" (builtins.readFile ../scripts/cudainfo.py);
  serial-wheel = ps:
    ps.buildPythonPackage {
      pname = "serial";
      version = "0.0.97";
      format = "wheel";

      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/1f/51/6a260c498162c37d0759f3759b7647a10d8d30caba1cfc9aa4b5b1f0d08b/serial-0.0.97-py2.py3-none-any.whl";
        sha256 = "e887f06e07e190e39174b694eee6724e3c48bd361be1d97964caef5d5b61c73b";
      };
    };
  navpy-wheel = ps:
    ps.buildPythonPackage {
      pname = "navpy";
      version = "1.0";
      format = "wheel";

      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/c5/3f/ca2d35641a60d4344a61acb9993416396b1ed039b90a03e34cb9ce47e47b/NavPy-1.0-py3-none-any.whl";
        sha256 = "6298f6793f748ecba966fb6faf212d2cc453129659595c8c92be513ca010df05";
      };
    };
  pyigrf-wheel = ps:
    ps.buildPythonPackage {
      pname = "pyigrf";
      version = "1.0.0";
      format = "wheel";

      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/3f/53/826e9f823b31120ddbd525070cd679e6bac6136aa5bb8ce4e112b6ddda34/pyigrf-1.0.0-py3-none-any.whl";
        sha256 = "733487d5750c8c2f8558c23d4bdebfb1e9c837e5aef437fc4425042c9dbb2d0b";
      };
    };
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
      pandas
      pygame
      pyopengl 
      scikit-learn
      scipy
      toml
      tomli
      pygobject3
      numba
      gprof2dot
      opencv4
      (serial-wheel ps)
      (navpy-wheel ps)
      (pyigrf-wheel ps)
      (onnxruntime-gpu-wheel ps)
      (tensorflow-wheel ps)
    ];
in {
  nixpkgs.config = {
    allowUnfree = true;
    cudaSupport = true;
    cudaCapabilities = ["7.2" "8.7"];
  };

  nixpkgs.overlays = [ opencvOverlay ];

  environment.variables = with pkgs; {
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      stdenv.cc.cc.lib
      cudaPackages.cudatoolkit
      cudaPackages.cudnn
      cudaPackages.tensorrt
      cudaPackages.vpi2
      nvidia-jetpack.l4t-cuda
      nvidia-jetpack.l4t-gstreamer
      nvidia-jetpack.l4t-multimedia
      nvidia-jetpack.l4t-camera
      blas
      lapack
      udev
      udev.dev
    ];
    NVCC_PREPEND_FLAGS = "--compiler-bindir ${pkgs.gcc11}/bin/gcc";
    NVCC_APPEND_FLAGS = "-I${pkgs.cudaPackages.cuda_cudart.include}/include";
    # Add environment variables to help CMake find BLAS/LAPACK
    BLAS_LIBRARY = "${pkgs.blas}/lib/libblas.so";
    LAPACK_LIBRARY = "${pkgs.lapack}/lib/liblapack.so";
    CMAKE_PREFIX_PATH = "${pkgs.blas}/lib:${pkgs.lapack}/lib";
  };
  hardware.graphics.enable = true;
  virtualisation.podman = {
    enable = true;
    # TODO: replace with `hardware.nvidia-container-toolkit.enable` when it works (https://github.com/nixos/nixpkgs/issues/344729).
    enableNvidia = true;
  };
  environment.systemPackages = with pkgs; [
    bash
    curl
    git
    jq
    udev
    udev.dev
    blas
    lapack
    tmux
    v4l-utils
    cmake
    bison
    flex
    fontforge
    makeWrapper
    gnumake
    libiconv
    autoconf
    automake
    libtool
    opencv4
    hdf5
    libjpeg8
    zip
    gfortran
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
  ];
}
