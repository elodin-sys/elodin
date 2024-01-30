{
  inputs = {
    elodin.url = "path:../../.";
    nixpkgs.follows = "elodin/nixpkgs";
    rust-overlay.follows = "elodin/rust-overlay";
    flake-utils.follows = "elodin/flake-utils";
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, crane, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [rust-overlay.overlays.default];
        };
        craneLib = (crane.mkLib pkgs).overrideToolchain pkgs.rust-bin.stable."1.73.0".default;
        crateName = craneLib.crateNameFromCargoToml { cargoToml = ./Cargo.toml; };
        src = pkgs.nix-gitignore.gitignoreSource [] ../..;
        commonArgs = {
          inherit (crateName) pname version;
          inherit src;
          doCheck = false;
          cargoExtraArgs = "--package=${crateName.pname}";
          buildInputs = with pkgs; [ protobuf ];
        };
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
        bin = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });

        dockerImage = pkgs.dockerTools.buildLayeredImage {
          name = "elo-atc";
          tag = "latest";
          contents = with pkgs; [cacert busybox];
          config = {
            Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
            Cmd = ["${bin}/bin/atc"];
          };
        };
      in
      {
        packages = {
          docker.image = dockerImage;
        };
      });
}
