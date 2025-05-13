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

  commonArgs = {
    inherit pname version;
    inherit src;
    doCheck = false;
    cargoExtraArgs = "--package=${pname}";
    HOST_CC = "${pkgs.stdenv.cc.nativePrefix}cc";
    TARGET_CC = "${pkgs.stdenv.cc.targetPrefix}cc";
    nativeBuildInputs = [pkg-config clang];
    buildInputs = [ffmpeg-full];
    LIBCLANG_PATH = "${pkgs.buildPackages.libclang.lib}/lib";
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  bin = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
    });
in
  bin
