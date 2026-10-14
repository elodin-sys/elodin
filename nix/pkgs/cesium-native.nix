{pkgs, ...}: let
  # The SDK build definition belongs to the same plugin revision used by Cargo.
  editorCargo = builtins.fromTOML (builtins.readFile ../../libs/elodin-editor/Cargo.toml);
  dependency = editorCargo.dependencies.bevy-cesium;
  plugin = builtins.fetchGit {
    url = dependency.git;
    rev = dependency.rev;
  };
in
  pkgs.callPackage "${plugin}/nix/cesium-native.nix" {}
