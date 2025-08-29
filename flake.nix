{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    crane.url = "github:ipetkov/crane";
    systems.url = "github:nix-systems/default";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
  };

  outputs = {
    self,
    nixpkgs,
    crane,
    rust-overlay,
    flake-utils,
    ...
  }: let
    rustToolchain = p: p.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
    elodinOverlay = final: prev: {
      elodin = {
        memserve = final.callPackage ./nix/pkgs/memserve.nix {inherit crane rustToolchain;};
        elodin-cli = final.callPackage ./nix/pkgs/elodin-cli.nix {inherit crane rustToolchain;};
        elodin-py = final.callPackage ./nix/pkgs/elodin-py.nix {inherit crane rustToolchain;};
        xla-ext = final.callPackage ./nix/pkgs/xla-ext.nix {inherit crane rustToolchain;};
      };
    };
  in
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [rust-overlay.overlays.default elodinOverlay];
        };
        config.packages = pkgs.elodin;
        docs-image = pkgs.callPackage ./nix/docs.nix {inherit config;};
        devShells = pkgs.callPackage ./nix/shell.nix {inherit config rustToolchain;};
      in {
        packages = with pkgs.elodin;
          {
            inherit
              memserve
              elodin-db
              elodin-cli
              elodin-py
              ;
          }
          // pkgs.lib.attrsets.optionalAttrs pkgs.stdenv.isLinux {
            inherit docs-image;
          };
        devShells = with devShells; {
          inherit
            c
            ops
            python
            nix-tools
            writing
            docs
            rust
            ;
        };
      }
    );
}
