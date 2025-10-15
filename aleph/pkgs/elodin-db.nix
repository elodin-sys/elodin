{
  pkgs,
  rustToolchain,
  lib,
  ...
}: let
  pname = "elodin-db";
  workspaceToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = workspaceToml.workspace.package.version;

  common = import ./common.nix {inherit lib;};
  src = common.src;

  # Main binary build
  bin = pkgs.rustPlatform.buildRustPackage {
    inherit pname version src;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "libs/db";

    nativeBuildInputs = [
      (rustToolchain pkgs)
    ];

    HOST_CC = "${pkgs.stdenv.cc.nativePrefix}cc";
    TARGET_CC = "${pkgs.stdenv.cc.targetPrefix}cc";

    doCheck = false;
  };

  # Clippy check derivation
  clippy = pkgs.rustPlatform.buildRustPackage {
    inherit version src;
    pname = "${pname}-clippy";

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "libs/db";

    nativeBuildInputs = [
      (rustToolchain pkgs)
    ];

    buildPhase = ''
      cargo clippy --package elodin-db --all-targets -- --deny warnings
    '';

    installPhase = "touch $out";
    doCheck = false;
  };

  # Test derivation using cargo-nextest
  test = pkgs.rustPlatform.buildRustPackage {
    inherit version src;
    pname = "${pname}-test";

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "libs/db";

    nativeBuildInputs = [
      (rustToolchain pkgs)
      pkgs.cargo-nextest
    ];

    buildPhase = ''
      cargo nextest run --package elodin-db --no-tests=pass
    '';

    installPhase = "touch $out";
    doCheck = false;
  };
in {
  inherit bin clippy test;
}
