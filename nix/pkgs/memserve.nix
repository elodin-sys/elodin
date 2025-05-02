{
  pkgs,
  crane,
  rustToolchain,
  ...
}: let
  craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
  commonArgs = {
    src = craneLib.cleanCargoSource ../../docs/memserve;
    doCheck = false;
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  bin = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
    });
in
  bin
