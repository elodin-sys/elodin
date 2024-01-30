{
  inputs = {
    elodin.url = "path:../../.";
    nixpkgs.follows = "elodin/nixpkgs";
    flake-utils.follows = "elodin/flake-utils";
    get-flake.follows = "elodin/get-flake";
  };

  outputs = inputs:
    with inputs;
      flake-utils.lib.eachDefaultSystem (
        system: let
          editor-web = get-flake ../../apps/editor-web;
          build_phoenix = pkgs: let
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
            src = ./.;
            version = "0.0.1";
          in
            beam_pkgs.mixRelease {
              inherit src version;
              pname = "elodin-dashboard";
              buildInputs = with pkgs; [nodePackages.tailwindcss];
              mixFodDeps = beam_pkgs.fetchMixDeps {
                inherit src version;
                pname = "mix-deps-dashboard";
                hash = "sha256-U7GzTATLDGbvt9WWecaSi2URryi0XXHhr9xCWaTBrV8";
              };
              ELODIN_TYPES_PATH = "./vendor/elodin_types";
              preConfigure = ''
                mkdir -p ./vendor/elodin_types
                cp --no-preserve=mode,ownership -r ${../../libs/elodin-types/elixir}/* ./vendor/elodin_types
              '';
              preBuild = ''
                mkdir -p ./priv/static/assets/wasm
                cp --no-preserve=mode,ownership -r ${editor-web.packages.${system}.default}/* ./priv/static/assets/wasm/
                ls -la ./priv/static/assets/wasm/
                cp -R --no-preserve=mode,ownership  ${./priv/static/images} ./priv/static/images
              '';
              postBuild = ''
                mix assets.deploy
              '';
            };
          build_docker = {
            bin,
            pkgs,
          }: let
            attrs = {
              name = "elo-dashboard";
              tag = "latest";
              contents = with pkgs; [cacert busybox];
              config = {
                Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
                Cmd = ["${bin}/bin/server"];
              };
            };
          in {
            image = pkgs.dockerTools.buildLayeredImage attrs;
            stream = pkgs.dockerTools.streamLayeredImage attrs;
          };
          pkgs = import nixpkgs {
            inherit system;
          };
        in rec {
          packages = {
            dashboard = build_phoenix pkgs;
            docker = build_docker {
              inherit pkgs;
              bin = packages.dashboard;
            };
          };
        }
      );
}
