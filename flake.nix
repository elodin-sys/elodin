{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/aba830385baac22d06a22085f5b5baeb88c88b46";
    cargo2nix.url = "github:cargo2nix/cargo2nix/release-0.11.0";
    rust-overlay.url = "github:oxalica/rust-overlay/b7a041430733fccaa1ffc3724bb9454289d0f701";
    flake-utils.url = "github:numtide/flake-utils";
    get-flake.url = "github:ursi/get-flake";
    nix2container.url = "github:nlewo/nix2container";
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
      packages.rust-overrides = pkgs:
        pkgs.rustBuilder.overrides.all
        ++ [
          (pkgs.rustBuilder.rustLib.makeOverride {
            name = "paracosm-types";
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
