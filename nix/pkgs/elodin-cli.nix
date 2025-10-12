{
  pkgs,
  crane,
  rustToolchain,
  lib,
  elodinPy,
  python,
  pythonPackages,
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
  pythonPath = pythonPackages.makePythonPath [elodinPy];
  pythonMajorMinor = lib.versions.majorMinor python.version;

  propagatedBuildInputs = with pkgs; [
    libGL
    libglvnd
    libxkbcommon
    wayland
    mesa
  ];

  bin = pkgs.rustPlatform.buildRustPackage rec {
    inherit pname version src propagatedBuildInputs;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "apps/elodin";

    nativeBuildInputs = with pkgs; [
      (rustToolchain pkgs)
      pkg-config
      cmake
      makeWrapper # Required for wrapProgram in postInstall
      gfortran # Fortran compiler needed at build time for netlib-src
    ];

    buildInputs = with pkgs;
      [
        python
      ]
      ++ lib.optionals stdenv.isDarwin [
        libiconv
        darwin.apple_sdk.frameworks.Security
        darwin.apple_sdk.frameworks.CoreServices
        darwin.apple_sdk.frameworks.SystemConfiguration
      ]
      ++ lib.optionals stdenv.isLinux [
        alsa-lib
        alsa-lib.dev
        udev
        libGL
      ];

    doCheck = false;

    postInstall =
      ''
        wrapProgram $out/bin/elodin \
          --prefix PATH : "${python}/bin" \
          --prefix PYTHONPATH : "${pythonPath}" \
          --prefix PYTHONPATH : "${python}/lib/python${pythonMajorMinor}" \
      ''
      + lib.optionalString pkgs.stdenv.isLinux ''
        --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath propagatedBuildInputs}
      '';

    # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
    # GCC 14 treats -Wincompatible-pointer-types as error by default
    NIX_CFLAGS_COMPILE = lib.optionalString pkgs.stdenv.isLinux "-Wno-error=incompatible-pointer-types";
    CARGO_PROFILE = "dev";
    CARGO_PROFILE_RELEASE_DEBUG = true;
  };
in {
  # Only export the binary - clippy and tests are run directly
  # via cargo in the development shell (nix develop)
  inherit bin;
}
