{
  nixConfig = {
    extra-substituters = ["https://s3-us-west-2.amazonaws.com/elodin-nix-cache"];
    extra-trusted-public-keys = [
      "elodin-cache-1:vvbmIQvTOjcBjIs8Ri7xlT2I3XAmeJyF5mNlWB+fIwM="
    ];
  };
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
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

    elodinOverlay = gitRev: final: prev: {
      tracy = (prev.tracy.override {withWayland = false;}).overrideAttrs (oldAttrs: {
        buildInputs =
          (oldAttrs.buildInputs or [])
          ++ (with prev; [libx11 libxrandr libxcursor libxi]);
      });
      elodin = rec {
        elodin-py = final.callPackage ./nix/pkgs/elodin-py.nix {
          inherit rustToolchain gitRev;
          python = final.python313;
          pythonPackages = final.python313Packages;
        };
        elodin-cli = final.callPackage ./nix/pkgs/elodin-cli.nix {
          inherit rustToolchain gitRev;
          elodinPy = elodin-py.py;
          python = elodin-py.python;
          pythonPackages = elodin-py.pythonPackages;
        };
        elodin-cli-tracy = final.callPackage ./nix/pkgs/elodin-cli.nix {
          inherit rustToolchain gitRev;
          elodinPy = elodin-py.py;
          python = elodin-py.python;
          pythonPackages = elodin-py.pythonPackages;
          enableTracy = true;
        };
        elodin-db = final.callPackage ./aleph/pkgs/elodin-db.nix {
          inherit rustToolchain gitRev;
        };
        elodinsink = final.callPackage ./nix/pkgs/elodinsink.nix {inherit rustToolchain;};
        rtsp-streamer = final.callPackage ./nix/pkgs/rtsp-streamer.nix {inherit rustToolchain;};
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
          config.allowUnfree = true;
          overlays = [
            rust-overlay.overlays.default
            (elodinOverlay gitRev)
          ];
        };

        config.packages = pkgs.elodin;

        shells = pkgs.callPackage ./nix/shell.nix {inherit config rustToolchain;};
      in {
        packages = with pkgs.elodin; {
          inherit elodin-cli elodin-db elodinsink rtsp-streamer;
          elodin-py = elodin-py.py;
        };

        devShells =
          (with shells; {
            inherit elodin;
            default = shells.elodin;
            run = pkgs.callPackage ./nix/run.nix {};
          })
          // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
            tracy = pkgs.callPackage ./nix/tracy.nix {};
          };

        formatter = pkgs.alejandra;
      }
    );
}
