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
    self,
    nixpkgs,
    flake-parts,
    rust-overlay,
    systems,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = import systems;
      perSystem = {
        config,
        self',
        inputs',
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
        wasm-bindgen-cli = {
          version = "0.2.97";
          hash = "sha256-DDUdJtjCrGxZV84QcytdxrmS5qvXD8Gcdq4OApj5ktI=";
          cargoHash = "sha256-Zfc2aqG7Qi44dY2Jz1MCdpcL3lk8C/3dt7QiE0QlNhc=";
        };
        imports = [
          ./nix/wasm-bindgen-cli.nix
          ./nix/shell.nix
          ./nix/atc.nix
          ./nix/editor-web.nix
          ./nix/dashboard.nix
          ./nix/docs.nix
          ./nix/elodin-cli.nix
          ./nix/elodin-py.nix
        ];
      };
    };
}
