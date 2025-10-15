{
  pkgs,
  rustToolchain,
  lib,
  gst_all_1,
  pkg-config,
  clang,
  ...
}: let
  pname = "gstelodin";
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
      cp $out/lib/libgstelodin.so $out/lib/gstreamer-1.0/
    '';
    
    doCheck = false;
  }
