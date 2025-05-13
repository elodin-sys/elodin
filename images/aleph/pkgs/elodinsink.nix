{
  pkgs,
  crane,
  rustToolchain,
  lib,
  gst_all_1,
  pkg-config,
  clang,
  ...
}: let
  craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
  pname = (craneLib.crateNameFromCargoToml {cargoToml = ../../../fsw/gstreamer/Cargo.toml;}).pname;
  version = (craneLib.crateNameFromCargoToml {cargoToml = ../../../Cargo.toml;}).version;

  common = import ./common.nix {inherit lib;};
  src = common.src;

  commonArgs = {
    inherit pname version;
    inherit src;
    doCheck = false;
    cargoExtraArgs = "--package=${pname}";
    HOST_CC = "${pkgs.stdenv.cc.nativePrefix}cc";
    TARGET_CC = "${pkgs.stdenv.cc.targetPrefix}cc";
    nativeBuildInputs = [pkg-config clang];
    buildInputs = [
      gst_all_1.gstreamer
      gst_all_1.gst-plugins-base
      gst_all_1.gst-plugins-good
    ];
    LIBCLANG_PATH = "${pkgs.buildPackages.libclang.lib}/lib";
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  bin = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
      postInstall = ''
        mkdir -p $out/lib/gstreamer-1.0
        cp $out/lib/libgstelodin.so $out/lib/gstreamer-1.0/
      '';
    });
in
  bin
