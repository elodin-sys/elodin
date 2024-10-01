{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane = {
      url = "github:ipetkov/crane";
    };
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ self, nixpkgs, flake-parts, rust-overlay, systems, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import systems;
      perSystem = { config, self', inputs', pkgs, system, ... }:
        let
          overlays = [
            rust-overlay.overlays.default
          ];
        in
        {
          _module.args = {
            pkgs = import nixpkgs {
              inherit system overlays;
              config.allowUnfree = true;
            };
            flakeInputs = inputs;
            rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
          };
          wasm-bindgen-cli = {
            version = "0.2.93";
            hash = "sha256-DDdu5mM3gneraM85pAepBXWn3TMofarVR4NbjMdz3r0=";
            cargoHash = "sha256-birrg+XABBHHKJxfTKAMSlmTVYLmnmqMDfRnmG6g/YQ=";
          };
          imports = [
            ./nix/wasm-bindgen-cli.nix
            ./nix/shell.nix
            ./nix/atc.nix
            ./nix/sim-agent.nix
            ./nix/editor-web.nix
            ./nix/dashboard.nix
            ./nix/docs.nix
            ./nix/elodin-cli.nix
            ./nix/sim-builder.nix
          ];
        };
    };
}
