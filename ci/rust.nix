{pkgs ? import <nixpkgs> {overlays = [(import (builtins.fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz"))];}}:
pkgs.mkShell.override {stdenv = pkgs.gcc12Stdenv;} {
  name = "paracosm-rust-shell";
  buildInputs = with pkgs; [
    rust-bin.stable.latest.default
    alsa-oss
    alsa-utils
    udev
    alsa-lib
    vulkan-loader
    libxkbcommon
    pkg-config
    libxkbcommon
    wayland
    fontconfig
    gtk3
    python3
    openssl
    clang
    protobuf
  ];
  LIBCLANG_PATH = "${pkgs.llvmPackages_14.libclang.lib}/lib";
  BINDGEN_EXTRA_CLANG_ARGS = with pkgs; ''${lib.optionalString stdenv.cc.isGNU "-isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config} -idirafter ${stdenv.cc.cc}/lib/gcc/${stdenv.hostPlatform.config}/${lib.getVersion stdenv.cc.cc}/include"}'';
  doCheck = false;
}
