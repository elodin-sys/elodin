{
  pkgs,
  config,
  ...
}: let
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
      Cmd = ["${config.packages.memserve}/bin/memserve" "--log-level" "debug"];
      WorkingDir = content;
    };
  };
in
  image
