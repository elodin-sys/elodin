{
  pkgs,
  crane,
  rustToolchain,
  lib,
  ffmpeg-full,
  pkg-config,
  clang,
  ...
}: let
  craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
  pname = (craneLib.crateNameFromCargoToml {cargoToml = ../../../fsw/video-streamer/Cargo.toml;}).pname;
  version = (craneLib.crateNameFromCargoToml {cargoToml = ../../../Cargo.toml;}).version;

  common = import ./common.nix {inherit lib;};
  src = common.src;

  commonArgs = with pkgs; {
    inherit pname version;
    inherit src;
    doCheck = false;
    cargoExtraArgs = "--package=${pname}";
    HOST_CC = "${stdenv.cc.nativePrefix}cc";
    TARGET_CC = "${stdenv.cc.targetPrefix}cc";
    nativeBuildInputs = [pkg-config clang cmake gfortran];
    buildInputs = [ffmpeg-full];
    LIBCLANG_PATH = "${buildPackages.libclang.lib}/lib";
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  bin = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
    });
in
  bin
