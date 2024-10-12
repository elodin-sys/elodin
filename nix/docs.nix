{ config, self', pkgs, lib, flakeInputs, ... }:
let
  content = pkgs.stdenv.mkDerivation {
    name = "docs-content";
    src = ../docs/public;

    buildInputs = [ pkgs.zola ];
    buildPhase = "zola build";

    installPhase = ''
      mkdir -p $out
      cp -r ./public $out/public
    '';
  };

  etag = pkgs.runCommand "content-etag" {} ''
    echo -n "${content}" | ${pkgs.b3sum}/bin/b3sum - | cut -d' ' -f1 | xargs printf "%s" > $out
  '';

  sws-config = pkgs.writeText "config.toml" ''
    [general]
    cache-control-headers = false
    port = 1111
    log-level = "info"

    [advanced]
    [[advanced.headers]]
    source = "**/*"
    [advanced.headers.headers]
    Cache-Control = "public, max-age=60"
    ETag = "${builtins.readFile etag}"
  '';


  image = pkgs.dockerTools.buildLayeredImage {
    name = "elo-docs";
    tag = "latest";
    config = {
      Env = [ "SERVER_CONFIG_FILE=${sws-config}" ];
      Cmd = [ "${pkgs.static-web-server}/bin/static-web-server" ];
      WorkingDir = content;
    };
  };
in
{
  packages.docs-sws-config = sws-config;
  packages.docs-content = content;
  packages.docs-image = image;
}
