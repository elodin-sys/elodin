{
  pkgs,
  crane,
  ...
}: let
  rustToolchain = pkgs.rust-bin.stable."1.84.0".default;
  craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
  crateName = craneLib.crateNameFromCargoToml {cargoToml = ../../../fsw/serial-bridge/Cargo.toml;};
  src = pkgs.nix-gitignore.gitignoreSource [] ../../../.;
  commonArgs = {
    inherit (crateName) pname version;
    inherit src;
    doCheck = false;
    cargoExtraArgs = "--package=${crateName.pname}";
    HOST_CC = "${pkgs.stdenv.cc.nativePrefix}cc";
    TARGET_CC = "${pkgs.stdenv.cc.targetPrefix}cc";
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  bin = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
    });
in
  bin
