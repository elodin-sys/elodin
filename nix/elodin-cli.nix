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
  crateName = craneLib.crateNameFromCargoToml {cargoToml = ../apps/elodin/Cargo.toml;};
  src = pkgs.nix-gitignore.gitignoreSource [] ../.;
  commonArgs = {
    inherit (crateName) pname version;
    inherit src;
    doCheck = false;
    cargoExtraArgs = "--package=${crateName.pname}";
    buildInputs = with pkgs; [
      protobuf
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
