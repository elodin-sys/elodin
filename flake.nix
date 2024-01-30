{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/aba830385baac22d06a22085f5b5baeb88c88b46";
    flake-utils.url = "github:numtide/flake-utils";
    get-flake.url = "github:ursi/get-flake";
    rust-overlay = {
      url = "github:oxalica/rust-overlay/e36f66bb10b09f5189dc3b1706948eaeb9a1c555";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
  };

  outputs = {
    nixpkgs,
    flake-utils,
    rust-overlay,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [rust-overlay.overlays.default];
        config.allowUnfree = true;
      };
    in rec {
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
