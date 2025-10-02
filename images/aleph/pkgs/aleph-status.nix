{
  pkgs,
  crane,
  rustToolchain,
  lib,
  ...
}: let
  craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
  crateName = craneLib.crateNameFromCargoToml {cargoToml = ../../../fsw/aleph-status/Cargo.toml;};
  version = (craneLib.crateNameFromCargoToml {cargoToml = ../../../Cargo.toml;}).version;

  common = import ./common.nix {inherit lib;};
  src = common.src;

  commonArgs = with pkgs; {
    inherit (crateName) pname;
    inherit src version;
    nativeBuildInputs = [cmake gfortran];
    doCheck = false;
    cargoExtraArgs = "--package=${crateName.pname}";
    HOST_CC = "${stdenv.cc.nativePrefix}cc";
    TARGET_CC = "${stdenv.cc.targetPrefix}cc";
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  bin = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
    });
in
  bin
