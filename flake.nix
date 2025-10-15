{
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
        memserve = final.callPackage ./nix/pkgs/memserve.nix {inherit rustToolchain;};
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
        # sensor-fw = final.callPackage ./nix/pkgs/sensor-fw.nix {inherit rustToolchain;};
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

        docs-image = pkgs.callPackage ./nix/docs.nix {inherit config;};
        shells = pkgs.callPackage ./nix/shell.nix {inherit config rustToolchain;};
      in {
        packages = with pkgs.elodin;
          {
            inherit
              memserve
              ;
            elodin-db = elodin-db.bin;
            elodin-cli = elodin-cli.bin;
            elodin-py = elodin-py.py;
            # sensor-fw = sensor-fw.bin;
            default = pkgs.elodin.elodin-cli.bin;
          }
          // pkgs.lib.attrsets.optionalAttrs pkgs.stdenv.isLinux {
            inherit docs-image;
          };

        checks = with pkgs.elodin; {
          elodin-db-clippy = elodin-db.clippy;
          # sensor-fw-clippy = sensor-fw.clippy;
          elodin-db-test = elodin-db.test;
          # sensor-fw-test = sensor-fw.test;
        };

        devShells = with shells; {
          inherit elodin;
          default = shells.elodin;
        };

        formatter = pkgs.alejandra;
      }
    );
}
