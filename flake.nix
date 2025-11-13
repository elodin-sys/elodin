{
  nixConfig = {
    extra-substituters = ["https://elodin-nix-cache.s3.us-west-2.amazonaws.com"];
    extra-trusted-public-keys = [
      "elodin-cache-1:vvbmIQvTOjcBjIs8Ri7xlT2I3XAmeJyF5mNlWB+fIwM="
    ];
    fallback = true;
  };
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
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
    rust-overlay,
    flake-utils,
    ...
  }: let
    rustToolchain = p: p.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
    elodinOverlay = final: prev: {
      elodin = rec {
        elodin-py = final.callPackage ./nix/pkgs/elodin-py.nix {
          inherit rustToolchain;
          python = final.python312Full;
          pythonPackages = final.python312Packages;
        };
        elodin-cli = final.callPackage ./nix/pkgs/elodin-cli.nix {
          inherit rustToolchain;
          elodinPy = elodin-py.py;
          python = elodin-py.python;
          pythonPackages = elodin-py.pythonPackages;
        };
        elodin-db = final.callPackage ./aleph/pkgs/elodin-db.nix {inherit rustToolchain;};
      };
    };
  in
    # overlays are system-agnostic ⇒ define them at top level
    {
      overlays.default = elodinOverlay;
    }
    // flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            rust-overlay.overlays.default
            elodinOverlay
          ];
        };

        config.packages = pkgs.elodin;

        shells = pkgs.callPackage ./nix/shell.nix {inherit config rustToolchain;};
      in {
        packages = with pkgs.elodin; {
          inherit elodin-cli elodin-db;
          elodin-py = elodin-py.py;
        };

        devShells = with shells; {
          inherit elodin;
          default = shells.elodin;
        };

        formatter = pkgs.alejandra;
      }
    );
}
