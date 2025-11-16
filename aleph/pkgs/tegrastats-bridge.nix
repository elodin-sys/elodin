{
  pkgs,
  rustToolchain,
  lib,
  ...
}: let
  pname = "tegrastats-bridge";
  workspaceToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = workspaceToml.workspace.package.version;

  common = import ./common.nix {inherit lib;};
  src = common.src;
in
  pkgs.rustPlatform.buildRustPackage {
    inherit pname version src;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "fsw/tegrastats-bridge";

    nativeBuildInputs = [
      (rustToolchain pkgs)
      pkgs.cmake
      pkgs.gfortran
    ];

    HOST_CC = "${pkgs.stdenv.cc.nativePrefix}cc";
    TARGET_CC = "${pkgs.stdenv.cc.targetPrefix}cc";

    doCheck = false;
  }
