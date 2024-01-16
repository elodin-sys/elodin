{
  inputs = {
    elodin.url = "path:../../.";
    nixpkgs.follows = "elodin/nixpkgs";
    rust-overlay.follows = "elodin/rust-overlay";
    flake-utils.follows = "elodin/flake-utils";
    nix2container.follows = "elodin/nix2container";
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs:
    with inputs;
      flake-utils.lib.eachDefaultSystem (
        system: let
          build_rust = pkgs: let
            xla_sha256_map = {
              "aarch64-darwin" = "sha256:0ykfnp6d78vp2yrhmr8wa3rlv6cri6mdl0fg034za839j7i7xqkz";
              "aarch64-linux" = "sha256:0sy53r6qhw0n3n342s013nq5rnzlg1qdbmgpvawh3p35a21qy8xr";
              "x86_64-linux"   = "sha256:103mybbnz6fm2i3r0fy0nf23ffdjxb37wd4pzvmwn0dpczr6dkw1";
            };
            xla_ext = fetchTarball {
            url = "https://github.com/elodin-sys/xla/releases/download/v0.5.4/xla_extension-${system}-gnu-cpu.tar.gz";
            sha256 = builtins.getAttr system xla_sha256_map;
            };
            craneLib = (crane.mkLib pkgs).overrideToolchain pkgs.rust-bin.stable."1.73.0".default;
            args = {
              pname = "elodin-web-editor";
              version = "0.0.1";
              src = pkgs.nix-gitignore.gitignoreSource [] ../..;
              doCheck = false;
              cargoExtraArgs = "--package=elodin-web-runner";
              buildInputs = with pkgs;
                [
                  systemdMinimal
                  alsa-lib
                  pkg-config
                  (python3.withPackages (ps: with ps; [numpy jax jaxlib]))
                  clang
                  protobuf
                  pango
                  gtk3
                ]
                ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
                  pkgs.libiconv
                ];
                XLA_EXTENSION_DIR = "${xla_ext}";
                LIBCLANG_PATH = "${pkgs.llvmPackages_14.libclang.lib}/lib";
                BINDGEN_EXTRA_CLANG_ARGS = with pkgs; ''${lib.optionalString stdenv.cc.isGNU "-isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config} -idirafter ${stdenv.cc.cc}/lib/gcc/${stdenv.hostPlatform.config}/${lib.getVersion stdenv.cc.cc}/include"}'';
            };
            cargoArtifacts =
              craneLib.buildDepsOnly args;
          in
            craneLib.buildPackage args
            // {
              inherit cargoArtifacts;
            };
          build_docker = {
            bin,
            pkgs,
          }: let
            args = {
              name = "elo-sim-runner";
              contents = with pkgs; [
                cacert
                busybox
                (python3.withPackages (ps: with ps; [numpy jax jaxlib]))
              ];
              tag = "latest";
              config = {
                Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
                Cmd = ["${bin}/bin/elodin-web-runner"];
              };
            };
          in {
            image = pkgs.dockerTools.buildLayeredImage args;
            stream = pkgs.dockerTools.streamLayeredImage args;
          };
          pkgs = import nixpkgs {
            inherit system;
            overlays = [rust-overlay.overlays.default];
          };
          aarch64_pkgs =
            if system == "aarch64-linux"
            then pkgs
            else
              import nixpkgs {
                localSystem = system;
                crossSystem = "aarch64-linux";
                overlays = [rust-overlay.overlays.default];
              };
          x86_64_pkgs =
            if system == "x86_64-linux"
            then pkgs
            else
              import nixpkgs {
                localSystem = system;
                crossSystem = "x86_64-linux";
                overlays = [rust-overlay.overlays.default];
              };
        in rec {
          packages = {
            elo-web-runner.default = build_rust pkgs;
            elo-web-runner.aarch64 = build_rust aarch64_pkgs;
            elo-web-runner.x86_64 = build_rust x86_64_pkgs;
            docker =
              (build_docker {
                inherit pkgs;
                bin = packages.elo-web-runner.default;
              })
              // {
                aarch64 = build_docker {
                  pkgs = aarch64_pkgs;
                  bin = packages.elo-web-runner.aarch64;
                };
                x86_64 = build_docker {
                  pkgs = x86_64_pkgs;
                  bin = packages.elo-web-runner.x86_64;
                };
              };
          };
        }
      );
}
