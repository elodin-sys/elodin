{ config, self', pkgs, lib, flakeInputs, ... }:
let
  nl2nix = import flakeInputs.npmlock2nix { inherit pkgs; };

  node_modules = nl2nix.v2.node_modules {
    src = ../docs/public;
    nodejs = pkgs.nodejs_21;
  };

  app = pkgs.stdenv.mkDerivation {
    name = "app";
    src = ../docs/public;

    buildInputs = [ node_modules ];

    installPhase = ''
      mkdir -p $out/app
      cp -r ${node_modules}/node_modules $out/app
      cp -r ./. $out/app
    '';
  };

  docker_attrs = {
    name = "elo-docs";
    tag = "latest";
    contents = [ pkgs.nodejs_21 app pkgs.cacert pkgs.busybox ];
    config = {
      Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
      Cmd = [ "node_modules/.bin/mintlify" "dev" ];
      ExposedPorts = { "3000/tcp" = {}; };
      WorkingDir = "/app";
    };
  };
in
{
  packages.docs-image = pkgs.dockerTools.buildLayeredImage docker_attrs;
}
