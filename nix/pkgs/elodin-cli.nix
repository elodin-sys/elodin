{
  pkgs,
  crane,
  rustToolchain,
  lib,
  ...
}: let
  # Direct Rust build using rustPlatform.buildRustPackage
  #
  # We bypass crane entirely due to path resolution issues with the pinned
  # crane version (dfd9a8dfd...) on macOS that cause "crane-utils" build failures.
  # For consistency and simplicity, we use the same direct build approach for
  # both macOS and Linux platforms.
  pname = "elodin";
  # Derive version from Cargo.toml workspace configuration
  cargoToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = cargoToml.workspace.package.version;
  src = pkgs.nix-gitignore.gitignoreSource [] ../..;

  bin = pkgs.rustPlatform.buildRustPackage rec {
    inherit pname version src;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "apps/elodin";

    nativeBuildInputs = with pkgs; [
      (rustToolchain pkgs)
      pkg-config
      cmake
    ];

    buildInputs = with pkgs;
      [
        python3
        gfortran
      ]
      ++ lib.optionals stdenv.isDarwin [
        libiconv
        darwin.apple_sdk.frameworks.Security
        darwin.apple_sdk.frameworks.CoreServices
        darwin.apple_sdk.frameworks.SystemConfiguration
      ]
      ++ lib.optionals stdenv.isLinux [
        alsa-lib
        udev
      ];

    doCheck = false;

    # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
    # GCC 14 treats -Wincompatible-pointer-types as error by default
    NIX_CFLAGS_COMPILE = lib.optionalString pkgs.stdenv.isLinux "-Wno-error=incompatible-pointer-types";
  };
in {
  # Only export the binary - clippy and tests are run directly
  # via cargo in the development shell (nix develop)
  inherit bin;
}
