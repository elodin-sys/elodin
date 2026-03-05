# Thin wrapper that imports the shared elodinsink package from nix/pkgs/
# This reduces code duplication between root and aleph flakes.
# NVIDIA-specific plugins (l4t-gstreamer) are added via the module, not the package.
{
  pkgs,
  rustToolchain,
  ...
}:
pkgs.callPackage ../../nix/pkgs/elodinsink.nix {
  inherit rustToolchain;
}
