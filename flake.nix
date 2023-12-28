{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/aba830385baac22d06a22085f5b5baeb88c88b46";
    flake-utils.url = "github:numtide/flake-utils";
    get-flake.url = "github:ursi/get-flake";
    nix2container = {
      url = "github:nlewo/nix2container";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay/b7a041430733fccaa1ffc3724bb9454289d0f701";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
    cargo2nix = {
      url = "github:cargo2nix/cargo2nix/release-0.11.0";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
        rust-overlay.follows = "rust-overlay";
      };
    };
  };

  outputs = {
    nixpkgs,
    flake-utils,
    cargo2nix,
    rust-overlay,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [cargo2nix.overlays.default rust-overlay.overlays.default];
        config.allowUnfree = true;
      };
      rustPkgs = pkgs.rustBuilder.makePackageSet {
        rustVersion = "1.73.0";
        packageFun = import ./Cargo.nix;
      };
    in rec {
      packages.starlark-json =
        (rustPkgs.workspace.starlark-json {}).bin;
      packages.buildkite-test-collector = pkgs.rustPlatform.buildRustPackage rec {
        pname = "buildkite-test-collector";
        version = "0.1.2";

        src = pkgs.fetchFromGitHub {
          owner = "buildkite";
          repo = "test-collector-rust";
          rev = version;
          hash = "sha256-ukBXUuy2rbyDWZD14Uf1AQQ7XiBB0jFHhcFmMOxV0V4";
        };
        postConfigure = ''
          cargo metadata --offline
        '';
        cargoHash = "sha256-6uMd+E95qlk/cVOzJwE5ZUaUsWkCmKLd3TbDSqIihic";
      };
      packages.wordchain = pkgs.gcc12Stdenv.mkDerivation rec {
        pname = "wordchain";
        version = "1.0.1";

        src = pkgs.fetchurl {
          url = "https://github.com/superorbital/wordchain/releases/download/v${version}/wordchain_linux_amd64";
          sha256 = "0039cd11ff90fba71de7bbab366bb43a3e1576766153a0ab153d69b1276a7e26";
        };
        phases = [ "installPhase" "patchPhase" ];
        installPhase = ''
          mkdir -p $out/bin
          cp $src $out/bin/wordchain
          chmod +x $out/bin/wordchain
        '';
      };
      devShells.rust = pkgs.mkShell.override {stdenv = pkgs.gcc12Stdenv;} {
        name = "elo-rust-shell";
        buildInputs = with pkgs;
          [
            rust-bin.stable.latest.default
            pkg-config
            python3
            openssl
            clang
            protobuf
            sccache
            packages.buildkite-test-collector
            alsa-lib
            alsa-oss
            alsa-utils
            vulkan-loader
            wayland
            gtk3
            udev
            libxkbcommon
            fontconfig
          ];
        LIBCLANG_PATH = "${pkgs.llvmPackages_14.libclang.lib}/lib";
        BINDGEN_EXTRA_CLANG_ARGS = with pkgs; ''${lib.optionalString stdenv.cc.isGNU "-isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config} -idirafter ${stdenv.cc.cc}/lib/gcc/${stdenv.hostPlatform.config}/${lib.getVersion stdenv.cc.cc}/include"}'';
        doCheck = false;
      };
      devShells.elixir = pkgs.mkShell.override {stdenv = pkgs.gcc12Stdenv;} {
        name = "elo-elixir-shell";
        buildInputs = with pkgs;
          [
            elixir
          ];
        doCheck = false;
      };
      devShells.ops = pkgs.mkShell.override {stdenv = pkgs.gcc12Stdenv;} {
        name = "elo-ops-shell";
        buildInputs = with pkgs;
          [
            skopeo
            gettext
            just
            _1password
            docker
            kubectl
            (google-cloud-sdk.withExtraComponents (with google-cloud-sdk.components; [gke-gcloud-auth-plugin]))
            packages.wordchain
          ];
        doCheck = false;
      };
      packages.rust-overrides = pkgs:
        pkgs.rustBuilder.overrides.all
        ++ [
          (pkgs.rustBuilder.rustLib.makeOverride {
            name = "elodin-types";
            overrideAttrs = drv: {
              propagatedNativeBuildInputs =
                drv.propagatedNativeBuildInputs
                or []
                ++ [
                  pkgs.buildPackages.protobuf
                ]
                ++ pkgs.lib.optional pkgs.buildPackages.hostPlatform.isDarwin [
                  pkgs.buildPackages.libiconv
                ];
            };
          })
          (pkgs.rustBuilder.rustLib.makeOverride {
            name = "ring";
            overrideAttrs = drv: {
              propagatedNativeBuildInputs =
                (drv.propagatedNativeBuildInputs
                  or [])
                ++ pkgs.lib.optional pkgs.buildPackages.hostPlatform.isDarwin [
                  pkgs.buildPackages.libiconv
                ];
            };
          })
          (pkgs.rustBuilder.rustLib.makeOverride {
            name = "libsqlite3-sys";
            overrideAttrs = drv: {
              propagatedNativeBuildInputs =
                (drv.propagatedNativeBuildInputs
                  or [])
                ++ pkgs.lib.optional pkgs.buildPackages.hostPlatform.isDarwin [
                  pkgs.buildPackages.libiconv
                ];
            };
          })
          (pkgs.rustBuilder.rustLib.makeOverride {
            name = "alsa-sys";
            overrideAttrs = drv: {
              buildInputs =
                (drv.buildInputs
                  or [])
                ++ [pkgs.alsa-lib];
            };
          })
        ];
    });
}
