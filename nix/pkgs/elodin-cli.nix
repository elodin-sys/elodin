{
  pkgs,
  crane,
  rustToolchain,
  lib,
  ...
}: let
  craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
  pname = (craneLib.crateNameFromCargoToml {cargoToml = ../../apps/elodin/Cargo.toml;}).pname;
  version = (craneLib.crateNameFromCargoToml {cargoToml = ../../Cargo.toml;}).version;
  src = pkgs.nix-gitignore.gitignoreSource [] ../../.;

  commonArgs = {
    inherit pname version;
    inherit src;
    doCheck = false;
    cargoExtraArgs = "--package=${pname}";
    buildInputs = with pkgs;
      [
        pkg-config
        python3
        cmake
        gfortran
      ]
      ++ lib.optionals stdenv.isLinux [
        alsa-lib
        udev
      ];
    # Pass through BUILDKITE environment variable
    BUILDKITE = builtins.getEnv "BUILDKITE";
  };

  cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  clippy = craneLib.cargoClippy (
    commonArgs
    // {
      inherit cargoArtifacts;
      cargoClippyExtraArgs = "--all-targets -- --deny warnings --allow deprecated";
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
