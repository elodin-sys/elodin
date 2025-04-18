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

  outputs = inputs @ {
    nixpkgs,
    flake-parts,
    rust-overlay,
    systems,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = import systems;
      perSystem = {
        pkgs,
        system,
        ...
      }: let
        overlays = [
          rust-overlay.overlays.default
        ];
      in {
        _module.args = {
          pkgs = import nixpkgs {
            inherit system overlays;
            config.allowUnfree = true;
          };
          flakeInputs = inputs;
          rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        };
        imports = [
          ./nix/shell.nix
          ./nix/docs.nix
          ./nix/elodin-cli.nix
          ./nix/elodin-py.nix
        ];
      };
    };
}
