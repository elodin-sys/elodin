{
  pkgs,
  crane,
  rustToolchain,
  lib,
  ...
}: let
  craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
  pname = (craneLib.crateNameFromCargoToml {cargoToml = ../../../libs/db/Cargo.toml;}).pname;
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
  };

  cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  clippy = craneLib.cargoClippy (
    commonArgs
    // {
      inherit cargoArtifacts;
      cargoClippyExtraArgs = "--all-targets -- --deny warnings";
    }
  );

  bin = craneLib.buildPackage (
    commonArgs
    // {
      inherit cargoArtifacts;
      doCheck = false;
    }
  );

  test = craneLib.cargoNextest (
    commonArgs
    // {
      inherit cargoArtifacts;
      partitions = 1;
      partitionType = "count";
      cargoNextestPartitionsExtraArgs = "--no-tests=pass";
    }
  );
in {
  inherit bin clippy test;
}
