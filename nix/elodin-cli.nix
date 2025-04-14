{
  config,
  self',
  pkgs,
  lib,
  flakeInputs,
  rustToolchain,
  ...
}: let
  craneLib = (flakeInputs.crane.mkLib pkgs).overrideToolchain rustToolchain;
  pname = (craneLib.crateNameFromCargoToml {cargoToml = ../apps/elodin/Cargo.toml;}).pname;
  version = (craneLib.crateNameFromCargoToml {cargoToml = ../Cargo.toml;}).version;
  src = pkgs.nix-gitignore.gitignoreSource [] ../.;
  commonArgs = {
    inherit pname version;
    inherit src;
    doCheck = false;
    cargoExtraArgs = "--package=${pname}";
    buildInputs = with pkgs; [
      pkg-config
      alsa-lib
      udev
    ];
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  bin = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
    });
in {
  packages.elodin-cli = bin;
}
