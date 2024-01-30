{
  inputs = {
    elodin.url = "path:../../.";
    nixpkgs.follows = "elodin/nixpkgs";
    flake-utils.follows = "elodin/flake-utils";

    npmlock2nix = {
      url = "github:nix-community/npmlock2nix/9197bbf397d76059a76310523d45df10d2e4ca81";
      flake = false;
    };
  };

  outputs = inputs:
    with inputs;
      flake-utils.lib.eachDefaultSystem (
        system: let

          pkgs = import nixpkgs {
            inherit system;
          };

          nl2nix = import npmlock2nix {
            inherit pkgs;
          };

          build_nodejs = pkgs.callPackage "${nixpkgs}/pkgs/development/web/nodejs/nodejs.nix" {
            python = pkgs.python3;
          };

          # NOTE: Mintlify dependency has a node version requirements ^18.17.0 || ^20.3.0 || >=21.0.0
          # NOTE: At the moment nodejs_20 is `20.2.0`, nodejs-18_x is `18.16.0`
          nodejs-20_5 = build_nodejs {
            enableNpm = true;
            version = "20.5.1";
            sha256 = "sha256-Q5xxqi84woYWV7+lOOmRkaVxJYBmy/1FSFhgScgTQZA=";
          };

          node_modules = nl2nix.v2.node_modules {
            src = ./.;
            nodejs = nodejs-20_5;
          };

          app = pkgs.stdenv.mkDerivation {
            name = "app";
            src = ./.;

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
            contents = [ nodejs-20_5 app pkgs.cacert pkgs.busybox ];
            config = {
              Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
              Cmd = [ "node_modules/.bin/mintlify" "dev" ];
              ExposedPorts = { "3000/tcp" = {}; };
              WorkingDir = "/app";
            };
          };
        in
        {
          packages = {
            docker = {
              image = pkgs.dockerTools.buildLayeredImage docker_attrs;
              stream = pkgs.dockerTools.buildLayeredImage docker_attrs;
            };
          };
        }
      );
  }
