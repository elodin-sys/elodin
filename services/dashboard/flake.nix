{
  inputs = {
    paracosm.url = "path:../../.";
    nixpkgs.follows = "paracosm/nixpkgs";
    flake-utils.follows = "paracosm/flake-utils";
    nix2container.follows = "paracosm/nix2container";
    get-flake.follows = "paracosm/get-flake";
  };

  outputs = inputs:
    with inputs;
      flake-utils.lib.eachDefaultSystem (
        system: let
          editor-web = get-flake ../../apps/editor-web;
          build_phoenix = pkgs: let
            beam = pkgs.beam.override {
              wxSupport = false;
              systemdSupport = false;
            };
            erlang = beam.beamLib.callErlang "${nixpkgs}/pkgs/development/interpreters/erlang/26.nix" {
              parallelBuild = true;
              wxSupport = false;
              systemdSupport = false;
              autoconf = pkgs.buildPackages.autoconf269;
            };
            beam_pkgs = beam.packagesWith erlang;
            src = ./.;
            version = "0.0.1";
          in
            beam_pkgs.mixRelease {
              inherit src version;
              pname = "paracosm-dashboard";
              buildInputs = with pkgs; [nodePackages.tailwindcss];
              mixFodDeps = beam_pkgs.fetchMixDeps {
                inherit src version;
                pname = "mix-deps-dashboard";
                hash = "sha256-yCo3MjVyDAd1SbtWBz3H31CnzGRLLWeKsUyugYEsuFA";
              };
              PARACOSM_TYPES_PATH = "./vendor/paracosm_types";
              preConfigure = ''
                mkdir -p ./vendor/paracosm_types
                cp --no-preserve=mode,ownership -r ${../../libs/paracosm-types/elixir}/* ./vendor/paracosm_types
              '';
              preBuild = ''
                mkdir -p ./priv/static/assets/wasm
                cp --no-preserve=mode,ownership -r ${editor-web.packages.${system}.default}/* ./priv/static/assets/wasm/
                ls -la ./priv/static/assets/wasm/
                cp -R --no-preserve=mode,ownership  ${./priv/static/images} ./priv/static/images
              '';
              postBuild = ''
                mix assets.deploy
                # for external task you need a workaround for the no deps check flag
                # https://github.com/phoenixframework/phoenix/issues/2690
                mix do deps.loadpaths --no-deps-check, phx.digest
                mix phx.digest --no-deps-check
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
