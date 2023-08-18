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
    xorg.libX11
    xorg.libXcursor
    xorg.libXi
    xorg.libXrandr # To use the x11 feature
    libxkbcommon
    pkg-config
  ];
}
