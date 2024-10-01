{ config, self', pkgs, lib, flakeInputs, ... }:
let
  app = pkgs.stdenv.mkDerivation {
    name = "app";
    src = ../docs/public;

    buildInputs = [ pkgs.zola ];
    buildPhase = "zola build";

    installPhase = ''
      mkdir -p $out/app
      cp -r ./public $out/app/public
    '';
  };

  docker_attrs = {
    name = "elo-docs";
    tag = "latest";
    contents = [ pkgs.static-web-server app pkgs.cacert pkgs.busybox ];
    config = {
      Env = [ "SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt" ];
      Cmd = [ "sh" "-c" "static-web-server -p 1111"];
      ExposedPorts = { "1111/tcp" = {}; };
      WorkingDir = "/app";
    };
  };
in
{
  packages.docs-image = pkgs.dockerTools.buildLayeredImage docker_attrs;
}
