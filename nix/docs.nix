{
  config,
  self',
  pkgs,
  lib,
  flakeInputs,
  ...
}: let
  content = pkgs.stdenv.mkDerivation {
    name = "docs-content";
    src = ../docs/public;

    buildInputs = [pkgs.zola];
    buildPhase = "zola build";

    installPhase = ''
      mkdir -p $out
      cp -r ./public $out/public
    '';
  };

  contentHash = lib.removeSuffix "\n" (
    builtins.readFile (
      pkgs.runCommand "content-hash" {} ''
        ${pkgs.nix}/bin/nix-hash --type sha256 --base32 ${content} > $out
      ''
    )
  );

  sws-config = pkgs.writeText "config.toml" ''
    [general]
    cache-control-headers = false
    port = 1111
    log-level = "info"

    [advanced]
    [[advanced.headers]]
    source = "**/*"
    [advanced.headers.headers]
    Cache-Control = "public, max-age=60, must-revalidate"
    ETag = "${contentHash}"
    Last-Modified = ""
  '';

  image = pkgs.dockerTools.buildLayeredImage {
    name = "elo-docs";
    tag = "latest";
    config = {
      Env = ["SERVER_CONFIG_FILE=${sws-config}"];
      Cmd = ["${pkgs.static-web-server}/bin/static-web-server"];
      WorkingDir = content;
    };
  };
in {
  packages.docs-sws-config = sws-config;
  packages.docs-content = content;
  packages.docs-image = image;
}
