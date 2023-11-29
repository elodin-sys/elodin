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
            beam_pkgs = with pkgs; beam.packagesWith beam.interpreters.erlang;
          in
            beam_pkgs.mixRelease {
              pname = "paracosm-dashboard";
              src = ./.;
              version = "0.0.1";
              buildInputs = with pkgs; [nodePackages.tailwindcss ];
              mixFodDeps = beam_pkgs.fetchMixDeps {
                src = ./.;
                version = "0.0.1";
                pname = "mix-deps-dashboard";
                hash = "sha256-XuwikAB2K8Rx5fTKWS3VFbo9ZPJ01sk+gGtI+3QyR6Y";
              };
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
          }:
            pkgs.dockerTools.buildLayeredImage {
              name = "dashboard";
              contents = with pkgs; [cacert busybox];
              config = {
                Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
                Cmd = ["${bin}/bin/server"];
              };
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
