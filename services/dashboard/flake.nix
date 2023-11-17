{
  inputs = {
    paracosm.url = "path:../../.";
    nixpkgs.follows = "paracosm/nixpkgs";
    flake-utils.follows = "paracosm/flake-utils";
    nix2container.follows = "paracosm/nix2container";
  };

  outputs = inputs:
    with inputs;
      flake-utils.lib.eachDefaultSystem (
        system: let
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
                hash = "sha256-SEFa/bBEiMWYZ1WsAg+S2EZ3Q+p1piZaPJELkzfKQIE";
              };
              postBuild = ''
                # mix assets.deploy
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
