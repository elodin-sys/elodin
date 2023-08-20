{pkgs ? import <nixpkgs> {overlays = [(import (builtins.fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz"))];}}:
pkgs.mkShell {
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
  ];
}
