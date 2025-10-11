{
  nixConfig = {
    extra-substituters = [
      "https://cache.nixos.org"
      "http://ci-arm1.elodin.dev:5000"
    ];
    extra-trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "builder-cache-1:q7rDGIQgkg1nsxNEg7mHN1kEDuxPmJhQpuIXCCwLj8E="
    ];
    fallback = true;
  };
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    # Pin crane to May 2025 version to avoid Cargo.lock path resolution bug on Linux
    # macOS python shell uses maturin directly instead (see nix/shell.nix)
    crane.url = "github:ipetkov/crane/dfd9a8dfd09db9aad544c4d3b6c47b12562544a5";
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
        elodin-db = final.callPackage ./images/aleph/pkgs/elodin-db.nix {inherit crane rustToolchain;};
        # sensor-fw = final.callPackage ./nix/pkgs/sensor-fw.nix {inherit crane rustToolchain;};
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
          elodin-cli-clippy = elodin-cli.clippy;
          elodin-py-clippy = elodin-py.clippy;
          # sensor-fw-clippy = sensor-fw.clippy;
          elodin-db-test = elodin-db.test;
          elodin-cli-test = elodin-cli.test;
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
