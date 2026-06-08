{
  pkgs,
  rustToolchain,
  lib,
  ffmpeg-headless,
  pkg-config,
  clang,
  ...
}: let
  pname = "video-streamer";
  workspaceToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = workspaceToml.workspace.package.version;

  common = import ./common.nix {inherit lib;};
  src = common.src;
  ffmpeg = ffmpeg-headless.override {
    withCuda = true;
  };
in
  pkgs.rustPlatform.buildRustPackage {
    inherit pname version src;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "fsw/video-streamer";

    nativeBuildInputs = [
      (rustToolchain pkgs)
      pkg-config
      clang
    ];

    buildInputs = [ffmpeg];

    HOST_CC = "${pkgs.stdenv.cc.nativePrefix}cc";
    TARGET_CC = "${pkgs.stdenv.cc.targetPrefix}cc";
    LIBCLANG_PATH = "${pkgs.buildPackages.libclang.lib}/lib";

    doCheck = false;
  }
