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
  craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
  pname = (craneLib.crateNameFromCargoToml {cargoToml = ../../apps/elodin/Cargo.toml;}).pname;
  version = (craneLib.crateNameFromCargoToml {cargoToml = ../../Cargo.toml;}).version;
  src = pkgs.nix-gitignore.gitignoreSource [] ../../.;
  pythonPath = pythonPackages.makePythonPath [elodinPy];
  pythonMajorMinor = lib.versions.majorMinor python.version;

  propagatedBuildInputs = with pkgs; [
    libGL
    libglvnd
    libxkbcommon
    wayland
    mesa
  ];

  commonArgs = with pkgs; {
    inherit pname version src propagatedBuildInputs;
    doCheck = false;
    cargoExtraArgs = "--package=${pname}";
    buildInputs =
      [
        pkg-config
        cmake
        gfortran
        python
      ]
      ++ lib.optionals stdenv.isLinux [
        alsa-lib
        alsa-lib.dev
        udev
        libGL
      ];

    nativeBuildInputs = with pkgs; [
      makeWrapper
      pkg-config
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
      // {
        postInstall =
          ''
            wrapProgram $out/bin/elodin \
              --prefix PATH : "${python}/bin" \
              --prefix PYTHONPATH : "${pythonPath}" \
              --prefix PYTHONPATH : "${python}/lib/python${pythonMajorMinor}" \
          ''
          + lib.optionalAttrs stdenv.isLinux ''
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
