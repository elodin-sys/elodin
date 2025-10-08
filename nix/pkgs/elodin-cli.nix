{
  pkgs,
  crane,
  rustToolchain,
  lib,
  ...
}: let
  craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
  pname = (craneLib.crateNameFromCargoToml {cargoToml = ../../apps/elodin/Cargo.toml;}).pname;
  version = (craneLib.crateNameFromCargoToml {cargoToml = ../../Cargo.toml;}).version;
  src = pkgs.nix-gitignore.gitignoreSource [] ../../.;

  propagatedBuildInputs = with pkgs; [
    libGL
    libglvnd
    libxkbcommon
    wayland
    mesa
  ];

  commonArgs = {
    inherit pname version src propagatedBuildInputs;
    doCheck = false;
    cargoExtraArgs = "--package=${pname}";
    buildInputs = with pkgs;
      [
        pkg-config
        python3
        cmake
        gfortran
        libGL
      ]
      ++ lib.optionals pkgs.stdenv.isLinux [
        alsa-lib
        alsa-lib.dev
        udev
      ];

    nativeBuildInputs = with pkgs;
      lib.optionals pkgs.stdenv.isLinux [
        pkg-config
        makeWrapper
      ];

    # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
    # GCC 14 treats -Wincompatible-pointer-types as error by default
    NIX_CFLAGS_COMPILE = lib.optionalString pkgs.stdenv.isLinux "-Wno-error=incompatible-pointer-types";
    CARGO_PROFILE = "dev";
    CARGO_PROFILE_RELEASE_DEBUG = true;
  };

  cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  clippy = craneLib.cargoClippy (
    commonArgs
    // {
      inherit cargoArtifacts;
      cargoClippyExtraArgs = "--all-targets -- --deny warnings --allow deprecated";
    }
  );

  bin = with pkgs;
    craneLib.buildPackage (
      commonArgs
      // {
        inherit cargoArtifacts;
        doCheck = false;
      }
      // lib.optionalAttrs stdenv.isLinux {
        postInstall = ''
          wrapProgram $out/bin/elodin \
            --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath propagatedBuildInputs}
        '';
        CARGO_PROFILE = "dev";
        CARGO_PROFILE_RELEASE_DEBUG = true;
        LD_LIBRARY_PATH = lib.makeLibraryPath propagatedBuildInputs;
      }
    );

  test = craneLib.cargoNextest (
    commonArgs
    // {
      inherit cargoArtifacts;
      partitions = 1;
      partitionType = "count";
      cargoNextestPartitionsExtraArgs = "--no-tests=pass";
    }
  );
in {
  inherit bin clippy test;
}
