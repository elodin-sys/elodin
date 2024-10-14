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
  crateName = craneLib.crateNameFromCargoToml {cargoToml = ../services/sim-builder/Cargo.toml;};
  protoFilter = path: _type: builtins.match ".*proto$" path != null;
  protoOrCargo = path: type: (protoFilter path type) || (craneLib.filterCargoSources path type);
  src = lib.cleanSourceWith {
    src = craneLib.path ./..;
    filter = protoOrCargo;
  };
  commonArgs = {
    inherit (crateName) pname version;
    inherit src;
    doCheck = false;
    cargoExtraArgs = "--package=${crateName.pname}";
    nativeBuildInputs = with pkgs; [protobuf python3];
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  bin = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
    });
in {
  packages.sim-builder = bin;
}
