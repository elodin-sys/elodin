{
  pkgs,
  rustToolchain,
  lib,
  ...
}: let
  pname = "mekf";
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

    buildAndTestSubdir = "fsw/mekf";

    nativeBuildInputs = [
      (rustToolchain pkgs)
    ];

    buildInputs = [
      pkgs.buildPackages.clang
    ];

    HOST_CC = "${pkgs.stdenv.cc.nativePrefix}cc";
    TARGET_CC = "${pkgs.stdenv.cc.targetPrefix}cc";
    LIBCLANG_PATH = "${pkgs.buildPackages.libclang.lib}/lib";

    doCheck = false;
  }
