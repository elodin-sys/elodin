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
      pkgs.cmake
      pkgs.gfortran
    ];

    HOST_CC = "${pkgs.stdenv.cc.nativePrefix}cc";
    TARGET_CC = "${pkgs.stdenv.cc.targetPrefix}cc";

    doCheck = false;
  };
in
  bin
