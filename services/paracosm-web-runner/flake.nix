{
  inputs = {
    paracosm.url = "path:../../.";
    nixpkgs.follows = "paracosm/nixpkgs";
    cargo2nix.follows = "paracosm/cargo2nix";
    rust-overlay.follows = "paracosm/rust-overlay";
    flake-utils.follows = "paracosm/flake-utils";
    nix2container.follows = "paracosm/nix2container";
    crane.url = "github:ipetkov/crane";
  };

  outputs = inputs:
    with inputs;
      flake-utils.lib.eachDefaultSystem (
        system: let
          build_rust = pkgs: let
            craneLib = (crane.mkLib pkgs).overrideToolchain pkgs.rust-bin.stable."1.73.0".default;
            args = {
              pname = "paracosm-web-editor";
              version = "0.0.1";
              src = pkgs.nix-gitignore.gitignoreSource [] ../..;
              doCheck = false;
              cargoExtraArgs = "--package=paracosm-web-runner";
              buildInputs = with pkgs;
                [
                  systemdMinimal
                  alsa-lib
                  pkg-config
                  (python3.withPackages (ps: with ps; [numpy]))
                  clang
                  protobuf
                ]
                ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
                  pkgs.libiconv
                ];
            };
            cargoArtifacts =
              craneLib.buildDepsOnly args;
          in
            craneLib.buildPackage args
            // {
              inherit cargoArtifacts;
            };
          build_docker = {
            bin,
            pkgs,
          }: let
            args = {
              name = "elo-web-runner";
              contents = with pkgs; [
                cacert
                busybox
                (python3.withPackages (ps: with ps; [numpy]))
              ];
              config = {
                Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
                Cmd = ["${bin}/bin/paracosm-web-runner"];
              };
            };
          in {
            image = pkgs.dockerTools.buildLayeredImage args;
            stream = pkgs.dockerTools.streamLayeredImage args;
          };
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
            elo-web-runner.default = build_rust pkgs;
            elo-web-runner.aarch64 = build_rust aarch64_pkgs;
            elo-web-runner.x86_64 = build_rust x86_64_pkgs;
            docker =
              (build_docker {
                inherit pkgs;
                bin = packages.elo-web-runner.default;
              })
              // {
                aarch64 = build_docker {
                  pkgs = aarch64_pkgs;
                  bin = packages.elo-web-runner.aarch64;
                };
                docker.x86_64 = build_docker {
                  pkgs = x86_64_pkgs;
                  bin = packages.elo-web-runner.x86_64;
                };
              };
          };
        }
      );
}
