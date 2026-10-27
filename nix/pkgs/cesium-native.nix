{pkgs, ...}: let
  # The SDK build definition belongs to the same plugin revision used by Cargo.
  workspaceCargo = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  dependency = workspaceCargo.workspace.dependencies.bevy-cesium;
  plugin = builtins.fetchGit {
    url = dependency.git;
    rev = dependency.rev;
  };
in
  pkgs.callPackage "${plugin}/nix/cesium-native.nix" {}
