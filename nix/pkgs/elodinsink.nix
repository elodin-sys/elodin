{
  pkgs,
  lib,
  rustToolchain,
  gst_all_1,
  pkg-config,
  clang,
  ...
}: let
  pname = "elodinsink";
  workspaceToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = workspaceToml.workspace.package.version;

  common = pkgs.callPackage ./common.nix {};
  src = common.src;

  # Cross-platform library extension
  libExt =
    if pkgs.stdenv.isDarwin
    then "dylib"
    else "so";
in
  pkgs.rustPlatform.buildRustPackage {
    inherit pname version src;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "fsw/gstreamer";

    nativeBuildInputs = [
      (rustToolchain pkgs)
      pkg-config
      clang
    ];

    buildInputs = with gst_all_1; [
      gstreamer
      gst-plugins-base
      gst-plugins-good
    ];

    HOST_CC = "${pkgs.stdenv.cc.nativePrefix}cc";
    TARGET_CC = "${pkgs.stdenv.cc.targetPrefix}cc";
    LIBCLANG_PATH = "${pkgs.buildPackages.libclang.lib}/lib";

    postInstall = ''
      mkdir -p $out/lib/gstreamer-1.0
      cp $out/lib/libgstelodin.${libExt} $out/lib/gstreamer-1.0/
    '';

    doCheck = false;

    meta = with lib; {
      description = "GStreamer plugin for streaming H.264 video to Elodin-DB";
      homepage = "https://github.com/elodin-sys/elodin";
      license = licenses.mit;
      mainProgram = "elodinsink";
    };
  }
