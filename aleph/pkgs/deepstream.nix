{
  dpkg,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  autoAddDriverRunpath,
  yaml_0_7,
  pkgs,
  ...
}:
stdenv.mkDerivation {
  pname = "deepstream";
  version = "7.1.0";
  src = fetchurl {
    url = "https://elo-public-misc.s3.us-east-2.amazonaws.com/deepstream-7.1_7.1.0-1_arm64.deb";
    sha256 = "8cc657e4784108c1a17da9bb8fbf736bc2ae4017065bb493486548c274465ca4";
  };
  nativeBuildInputs = [dpkg autoPatchelfHook autoAddDriverRunpath];

  buildInputs = [
    stdenv.cc.cc.lib

    pkgs.gst_all_1.gst-rtsp-server
    pkgs.nvidia-jetpack.cudaPackages.cudatoolkit
    pkgs.nvidia-jetpack.cudaPackages.cudnn
    pkgs.nvidia-jetpack.cudaPackages.tensorrt
    pkgs.nvidia-jetpack.cudaPackages.vpi
    pkgs.nvidia-jetpack.l4t-core
    pkgs.nvidia-jetpack.l4t-multimedia
    pkgs.nvidia-jetpack.l4t-gstreamer
    pkgs.json-glib
    pkgs.jsoncpp
    yaml_0_7
    pkgs.protobuf
    pkgs.grpc
  ];

  unpackCmd = "dpkg-deb -x $src source";

  autoPatchelfIgnoreMissingDeps = true;

  preBuild = ''

    # Add NVIDIA JetPack library paths
    addAutoPatchelfSearchPath ${pkgs.nvidia-jetpack.l4t-core}/lib
    addAutoPatchelfSearchPath ${pkgs.nvidia-jetpack.l4t-multimedia}/lib
    # Add gst-rtsp-server library path for deepstream-app
    addAutoPatchelfSearchPath ${pkgs.gst_all_1.gst-rtsp-server}/lib
  '';

  # After copying files, we need to ensure DeepStream's own libraries can find each other
  postUnpack = ''
    # Add DeepStream's own lib directory to autoPatchelf search path
    # This needs to happen after unpack but before patching
    if [ -d "source/opt/nvidia/deepstream/deepstream-7.1/lib" ]; then
      addAutoPatchelfSearchPath "$(pwd)/source/opt/nvidia/deepstream/deepstream-7.1/lib"
    fi
  '';

  postPatch = ''
    ls -la
    cp -r opt/nvidia/deepstream/deepstream-7.1/* .

    # Remove all sample apps BEFORE removing opt directory to avoid broken symlinks
    rm -rf sources/apps/sample_apps
    rm -rf sources/objectDetector_FasterRCNN
    rm -rf sources/apps/triton
    # Remove sample binaries from bin directory
    find bin -name "deepstream-test*" -delete || true
    find bin -name "deepstream-*-app" -delete || true
    find bin -name "deepstream-3d-*" -delete || true
    # Clean up any remaining sample-related directories or files
    find sources -type d -name "*sample*" -exec rm -rf {} + || true

    # Also remove from original opt structure before we delete it
    rm -rf opt/nvidia/deepstream/deepstream-7.1/sources/apps/sample_apps || true
    rm -rf opt/nvidia/deepstream/deepstream-7.1/sources/objectDetector_FasterRCNN || true
    rm -rf opt/nvidia/deepstream/deepstream-7.1/sources/apps/triton || true

    # Now remove the entire opt directory since we've copied what we need
    rm -rf opt
    mv lib/gst-plugins lib/gstreamer-1.0

    # we don't want to bring in azure crap
    rm lib/libnvds_azure_edge_proto.so
    rm lib/libiothub_client.so

    # no kafka
    rm lib/libnvds_kafka_proto.so

    # Final cleanup: remove any broken symlinks that might remain
    find . -type l ! -exec test -e {} \; -delete || true

  '';

  installPhase = ''
    runHook preInstall

    cp -r . $out

    runHook postInstall
  '';
}
