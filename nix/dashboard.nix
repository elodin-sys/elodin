{ config, self', pkgs, lib, ... }:
let
  beam = pkgs.beam;
  # Note the below code disables wx widgets and systemd, which drastically reduces image size.
  # It causes erlang to be fully rebuilt, so it is commented out for now.
  # beam = pkgs.beam.override {
  #   wxSupport = false;
  #   systemdSupport = false;
  # };
  # erlang = beam.beamLib.callErlang "${nixpkgs}/pkgs/development/interpreters/erlang/26.nix" {
  #   parallelBuild = true;
  #   wxSupport = false;
  #   systemdSupport = false;
  #   autoconf = pkgs.buildPackages.autoconf269;
  # };
  beam_pkgs = beam.packagesWith beam.interpreters.erlang;
  src = ../services/dashboard;
  version = "0.0.1";
  bin = beam_pkgs.mixRelease {
    inherit src version;
    pname = "elodin-dashboard";
    buildInputs = with pkgs; [nodePackages.tailwindcss];
    mixFodDeps = beam_pkgs.fetchMixDeps {
      inherit src version;
      pname = "mix-deps-dashboard";
      hash = "sha256-UX6tSUTgEN1KiuiNed63wffmIYx0HnQgkqZlD2uWRO8=";
    };
    ELODIN_TYPES_PATH = "./vendor/elodin_types";
    preConfigure = ''
      mkdir -p ./vendor/elodin_types
      cp --no-preserve=mode,ownership -r ${../libs/elodin-types/elixir}/* ./vendor/elodin_types
    '';
    preBuild = ''
      mkdir -p ./priv/static/assets/wasm
      cp --no-preserve=mode,ownership -r ${config.packages.editor-web}/* ./priv/static/assets/wasm/
      ls -la ./priv/static/assets/wasm/
      cp -R --no-preserve=mode,ownership  ${../services/dashboard/priv/static/images} ./priv/static/images
    '';
    postBuild = ''
      mix ua_inspector.download --force
      mix assets.deploy
    '';
  };

  image = pkgs.dockerTools.buildLayeredImage {
    name = "elo-dashboard";
    tag = "latest";
    contents = with pkgs; [cacert busybox];
    config = {
      Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
      Cmd = ["${bin}/bin/server"];
    };
  };
in
{
  packages.dashboard-image = image;
}
