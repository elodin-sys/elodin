{
  inputs = {
    paracosm.url = "path:../../.";
    nixpkgs.follows = "paracosm/nixpkgs";
    cargo2nix.follows = "paracosm/cargo2nix";
    rust-overlay.follows = "paracosm/rust-overlay";
    flake-utils.follows = "paracosm/flake-utils";
  };

  outputs = inputs:
    with inputs; let
      build_rust = pkgs: let
        rustPkgs = pkgs.rustBuilder.makePackageSet {
          rustVersion = "1.73.0";
          packageFun = import ../../Cargo.nix;
          packageOverrides = paracosm.packages.rust-overrides;
        };
      in
        (rustPkgs.workspace.atc {}).bin;
      build_docker = {
        bin,
        pkgs,
      }:
        pkgs.dockerTools.buildLayeredImage {
          name = "atc";
          contents = [
            pkgs.cacert
          ];
          config = {
            Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
            Cmd = ["${bin.bin}/bin/atc"];
          };
        };
    in
      flake-utils.lib.eachDefaultSystem (
        system: let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [cargo2nix.overlays.default rust-overlay.overlays.default];
          };
          aarch64_pkgs =
            if system == "aarch64-linux"
            then pkgs
            else
              import nixpkgs {
                localSystem = system;
                crossSystem = "aarch64-linux";
                overlays = [cargo2nix.overlays.default rust-overlay.overlays.default];
              };
          x86_64_pkgs =
            if system == "x86_64-linux"
            then pkgs
            else
              import nixpkgs {
                localSystem = system;
                crossSystem = "x86_64-linux";
                overlays = [cargo2nix.overlays.default rust-overlay.overlays.default];
              };
        in rec {
          packages = {
            atc.default = build_rust pkgs;
            atc.aarch64 = build_rust aarch64_pkgs;
            atc.x86_64 = build_rust x86_64_pkgs;
            docker.default = build_docker {
              inherit pkgs;
              bin = packages.atc.default;
            };
            docker.aarch64 = build_docker {
              pkgs = aarch64_pkgs;
              bin = packages.atc.aarch64;
            };
            docker.x86_64 = build_docker {
              pkgs = x86_64_pkgs;
              bin = packages.atc.x86_64;
            };
          };
        }
      );
}
