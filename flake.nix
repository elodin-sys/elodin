{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/aba830385baac22d06a22085f5b5baeb88c88b46";
    systems.url = "github:nix-systems/default";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay/e36f66bb10b09f5189dc3b1706948eaeb9a1c555";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    npmlock2nix = {
      url = "github:nix-community/npmlock2nix/9197bbf397d76059a76310523d45df10d2e4ca81";
      flake = false;
    };
  };

  outputs = inputs@{ self, nixpkgs, flake-parts, rust-overlay, systems, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import systems;
      perSystem = { config, self', inputs', pkgs, system, ... }:
        let
          overlays = [ (import rust-overlay) ];
        in
        {
          _module.args = {
            pkgs = import nixpkgs {
              inherit system overlays;
              config.allowUnfree = true;
            };
            flakeInputs = inputs;
            rustToolchain = pkgs.rust-bin.stable.latest.default;
          };
          buildkite-test-collector = {
            version = "0.1.2";
            hash = "sha256-ukBXUuy2rbyDWZD14Uf1AQQ7XiBB0jFHhcFmMOxV0V4";
            cargoHash = "sha256-6uMd+E95qlk/cVOzJwE5ZUaUsWkCmKLd3TbDSqIihic";
          };
          imports = [
            ./nix/buildkite.nix
            ./nix/shell.nix
            ./nix/atc.nix
            ./nix/sim-agent.nix
            ./nix/editor-web.nix
            ./nix/dashboard.nix
            ./nix/docs.nix
          ];
        };
    };
}
