{
  pkgs,
  flakeInputs,
  rustToolchain,
  ...
}: let
  craneLib = (flakeInputs.crane.mkLib pkgs).overrideToolchain rustToolchain;
  commonArgs = {
    src = craneLib.cleanCargoSource ../docs/memserve;
    doCheck = false;
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  memserve = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
    });

  content = pkgs.stdenv.mkDerivation {
    name = "docs-content";
    src = ../docs/public;

    buildInputs = [pkgs.zola];
    buildPhase = "zola build";

    installPhase = ''
      mkdir -p $out
      cp -r ./public/* $out/
    '';
  };

  image = pkgs.dockerTools.buildLayeredImage {
    name = "elo-docs";
    tag = "latest";
    config = {
      Cmd = ["${memserve}/bin/memserve" "--log-level" "debug"];
      WorkingDir = content;
    };
  };
in {
  packages.memserve = memserve;
  packages.docs-content = content;
  packages.docs-image = image;
}
