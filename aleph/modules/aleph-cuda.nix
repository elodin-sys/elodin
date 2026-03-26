{
  lib,
  pkgs,
  ...
}: let
  cudaToolkit = pkgs.cudaPackages.cudatoolkit;
  cudnn = pkgs.cudaPackages.cudnn;
  tensorrt = pkgs.nvidia-jetpack.cudaPackages.tensorrt;
  cudaLibraryPaths = with pkgs;
    map (p: "${lib.getOutput "lib" p}/lib") [
      stdenv.cc.cc.lib
      cudaToolkit
      cudnn
      tensorrt
      deepstream
      jsoncpp
      nvidia-jetpack.l4t-cuda
      nvidia-jetpack.l4t-gstreamer
      nvidia-jetpack.l4t-multimedia
      nvidia-jetpack.l4t-camera
    ];
  cudaIncludePaths = [
    "${cudaToolkit}/include"
    "${cudnn}/include"
  ];
  nvccFlags = lib.concatStringsSep " " [
    "--compiler-bindir ${pkgs.gcc13}/bin/gcc"
    "-I${cudaToolkit}/include"
    "-I${cudnn}/include"
  ];
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
  gpuPythonPackages = ps:
    with ps; [
      onnx
      pycuda
      (onnxruntime-gpu-wheel ps)
    ];
in {
  config = {
    aleph.dev.extraPythonPackageSets = [gpuPythonPackages];

    nixpkgs.config = {
      cudaSupport = true;
      cudaCapabilities = ["8.7"];
    };

    hardware.graphics.enable = true;
    hardware.nvidia-container-toolkit.enable = true;

    environment.variables = {
      CUDA_HOME = "${cudaToolkit}";
      CUDA_PATH = "${cudaToolkit}";
      CUDA_ROOT = "${cudaToolkit}";
      CUDAToolkit_ROOT = "${cudaToolkit}";
      CUPY_INCLUDE_PATH = "${cudaToolkit}/include";
      CPATH = cudaIncludePaths;
      C_INCLUDE_PATH = cudaIncludePaths;
      CPLUS_INCLUDE_PATH = cudaIncludePaths;
      LIBRARY_PATH = cudaLibraryPaths;
      LD_LIBRARY_PATH = cudaLibraryPaths;
      NVCC_PREPEND_FLAGS = nvccFlags;
      GST_PLUGIN_PATH = with pkgs;
        map (p: "${lib.getOutput "lib" p}/lib/gstreamer-1.0") [
          deepstream
          nvidia-jetpack.l4t-gstreamer
        ];
    };

    environment.systemPackages = with pkgs; [
      cudainfo
      cudaToolkit
      cudnn
      tensorrt
      nvidia-jetpack.l4t-gstreamer
      nvidia-jetpack.l4t-multimedia
      cudaPackages.cuda_nvcc
      nvidia-jetpack.samples.cuda-test
      nvidia-jetpack.samples.cudnn-test
      deepstream
    ];
  };
}
