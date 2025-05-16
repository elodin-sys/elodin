{
  dpkg,
  stdenv,
  lib,
  autoPatchelfHook,
  autoAddDriverRunpath,
  gst_all_1,
  cudaPackages,
  json-glib,
  jsoncpp,
  yaml-cpp,
  protobuf,
  grpc,
  avahi,
  openssl,
  yaml_0_6,
  ...
}:
stdenv.mkDerivation {
  name = "deepstream";
  src = ../vendor/deepstream-6.3_6.3.0-1_arm64.deb;
  nativeBuildInputs = [dpkg autoPatchelfHook autoAddDriverRunpath];
  buildInputs = [
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    cudaPackages.tensorrt
    cudaPackages.vpi2
    stdenv.cc.cc.lib
    gst_all_1.gstreamer
    gst_all_1.gst-plugins-base
    gst_all_1.gst-rtsp-server
    json-glib
    jsoncpp
    yaml_0_6
    protobuf
    grpc
    avahi
    openssl
  ];

  unpackCmd = "for src in $srcs; do dpkg-deb -x $src source; done";

  autoPatchelfIgnoreMissingDeps = true;

  preBuild = ''
    addAutoPatchelfSearchPath ${yaml-cpp}/lib/
    addAutoPatchelfSearchPath ${jsoncpp}/lib/
  '';


  postPatch = ''
    ls -la
    cp -r opt/nvidia/deepstream/deepstream-6.3/* .
    rm -rf nvidia
    mv lib/gst-plugins lib/gstreamer-1.0

    # we don't want to bring in azure crap
    rm lib/libnvds_azure_edge_proto.so
    rm lib/libiothub_client.so

    # no kafka
    rm lib/libnvds_kafka_proto.so

  '';

  installPhase = ''
    runHook preInstall

    cp -r . $out

    runHook postInstall
  '';
}
