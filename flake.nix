{
  nixConfig = {
    extra-substituters = ["https://elodin-nix-cache.s3.us-west-2.amazonaws.com"];
    extra-trusted-public-keys = [
      "elodin-cache-1:vvbmIQvTOjcBjIs8Ri7xlT2I3XAmeJyF5mNlWB+fIwM="
    ];
  };
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
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
    tracySrc = p:
      p.fetchFromGitHub {
        owner = "wolfpld";
        repo = "tracy";
        rev = "5479a42ef9346b64e6d1b860ae58aa8abdb0c7f6";
        hash = "sha256-4J8b+72k+xpeT6KsrkioF1xfWEBsGg2eLRg9iONxP/I=";
      };

    elodinOverlay = gitRev: final: prev: {
      tracy = final.callPackage ./nix/pkgs/tracy.nix {
        tracy = prev.tracy;
      };
      elodin = rec {
        iree_runtime_tracy =
          if final.stdenv.isLinux
          then
            final.callPackage ./nix/pkgs/iree-runtime.nix {
              enableTracing = true;
              tracySrc = tracySrc final;
            }
          else null;

        elodin-py = final.callPackage ./nix/pkgs/elodin-py.nix {
          inherit rustToolchain;
          python = final.python313;
          pythonPackages = final.python313Packages;
        };
        elodin-py-tracy = final.callPackage ./nix/pkgs/elodin-py.nix {
          inherit rustToolchain iree_runtime_tracy;
          python = final.python313;
          pythonPackages = final.python313Packages;
          enableTracy = true;
        };
        elodin-cli = final.callPackage ./nix/pkgs/elodin-cli.nix {
          inherit rustToolchain gitRev;
          elodinPy = elodin-py.py;
          python = elodin-py.python;
          pythonPackages = elodin-py.pythonPackages;
        };
        elodin-cli-tracy = final.callPackage ./nix/pkgs/elodin-cli.nix {
          inherit rustToolchain gitRev;
          elodinPy = elodin-py-tracy.py;
          python = elodin-py-tracy.python;
          pythonPackages = elodin-py-tracy.pythonPackages;
          enableTracy = true;
        };
        elodin-db = final.callPackage ./aleph/pkgs/elodin-db.nix {
          inherit rustToolchain gitRev;
        };
        elodinsink = final.callPackage ./nix/pkgs/elodinsink.nix {inherit rustToolchain;};
      };
    };
  in
    # overlays are system-agnostic ⇒ define them at top level
    {
      overlays.default = elodinOverlay "unknown";
    }
    // flake-utils.lib.eachDefaultSystem (
      system: let
        gitRev =
          nixpkgs.lib.substring 0 7
          (self.shortRev or self.dirtyShortRev or self.rev or self.dirtyRev or "unknown");
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            rust-overlay.overlays.default
            (elodinOverlay gitRev)
          ];
        };

        config.packages = pkgs.elodin;

        shells = pkgs.callPackage ./nix/shell.nix {inherit config rustToolchain;};
      in {
        packages = with pkgs.elodin; {
          inherit elodin-cli elodin-db elodinsink;
          elodin-py = elodin-py.py;
        };

        devShells = with shells; {
          inherit elodin;
          default = shells.elodin;
          run = pkgs.callPackage ./nix/run.nix {};
          tracy = pkgs.callPackage ./nix/tracy.nix {};
        };

        formatter = pkgs.alejandra;
      }
    );
}
