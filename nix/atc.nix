{
  config,
  self',
  pkgs,
  lib,
  flakeInputs,
  rustToolchain,
  ...
}: let
  craneLib = (flakeInputs.crane.mkLib pkgs).overrideToolchain rustToolchain;
  crateName = craneLib.crateNameFromCargoToml {cargoToml = ../services/atc/Cargo.toml;};
  src = pkgs.nix-gitignore.gitignoreSource [] ../.;
  commonArgs = {
    inherit (crateName) pname version;
    inherit src;
    doCheck = false;
    cargoExtraArgs = "--package=${crateName.pname}";
    buildInputs = with pkgs; [protobuf openssl];
    OPENSSL_DIR = "${pkgs.openssl.dev}";
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
    OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include/";
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  bin = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
    });

  image = pkgs.dockerTools.buildLayeredImage {
    name = "elo-atc";
    tag = "latest";
    contents = with pkgs; [cacert busybox];
    config = {
      Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
      Cmd = ["${bin}/bin/${crateName.pname}"];
    };
  };
in {
  packages.atc-image = image;
}
