{
  pkgs,
  lib,
  crane,
  rustToolchain,
  system,
  ...
}: let
  xla_ext = pkgs.callPackage ./xla-ext.nix {inherit system;};
  craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
  pname = (craneLib.crateNameFromCargoToml {cargoToml = ../../libs/nox-py/Cargo.toml;}).pname;
  version = (craneLib.crateNameFromCargoToml {cargoToml = ../../Cargo.toml;}).version;

  src = pkgs.nix-gitignore.gitignoreSource [] ../..;

  arch = with pkgs;
    if stdenv.isDarwin
    then
      # Python wheels require "arm64" for ARM Macs, not "aarch64"
      if stdenv.hostPlatform.ubootArch == "aarch64"
      then "arm64"
      else stdenv.hostPlatform.ubootArch
    else builtins.elemAt (lib.strings.splitString "-" system) 0;

  commonArgs = {
    inherit pname version;
    inherit src;
    doCheck = false;

    nativeBuildInputs = with pkgs;
      [maturin]
      ++ lib.optionals stdenv.isLinux [
        autoPatchelfHook
        patchelf
      ]
      ++ lib.optionals stdenv.isDarwin [
        fixDarwinDylibNames
        darwin.cctools
      ];
    buildInputs = with pkgs;
      [
        pkg-config
        python3
        openssl
        cmake
        gfortran
        gfortran.cc.lib
        xla_ext
      ]
      ++ lib.optionals stdenv.isDarwin [pkgs.libiconv];

    cargoExtraArgs = "--package=nox-py";

    XLA_EXTENSION_DIR = "${xla_ext}";
    OPENSSL_DIR = "${pkgs.openssl.dev}";
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
    OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include/";

    # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
    # GCC 14 treats -Wincompatible-pointer-types as error by default
    NIX_CFLAGS_COMPILE = lib.optionalString pkgs.stdenv.isLinux "-Wno-error=incompatible-pointer-types";
  };

  cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  clippy = craneLib.cargoClippy (
    commonArgs
    // {
      inherit cargoArtifacts;
      cargoClippyExtraArgs = "--all-targets -- --deny warnings";
    }
  );

  wheelName = "elodin";
  wheelPlatform =
    if pkgs.stdenv.isDarwin
    then "macosx_11_0"
    else "linux";
  wheelSuffix = "cp310-abi3-${wheelPlatform}_${arch}";
  # Convert version format from 0.15.0-alpha.1 to 0.15.0a1 for wheel filename
  wheelVersion = lib.strings.replaceStrings ["-alpha."] ["a"] version;
  wheel = craneLib.buildPackage (
    commonArgs
    // {
      inherit cargoArtifacts;
      doCheck = false;
      pname = "elodin";
      buildPhase = "maturin build --offline --target-dir ./target -m libs/nox-py/Cargo.toml --release";
      installPhase = "install -D target/wheels/${wheelName}-${wheelVersion}-${wheelSuffix}.whl -t $out/";
    }
  );

  elodin = ps: let
    # Create a modified Python package set with our JAX/jaxlib overrides
    # This ensures all packages use the same jaxlib version
    ps' = ps.override {
      overrides = self: super: {
        # Override jaxlib globally in this package set
        jaxlib = super.jaxlib-bin.overridePythonAttrs (old: rec {
          version = "0.4.31";
          src = let
            system = pkgs.stdenv.system;
            platform =
              if system == "x86_64-linux"
              then "manylinux2014_x86_64"
              else if system == "aarch64-linux"
              then "manylinux2014_aarch64"
              else "macosx_11_0_arm64";
            wheelName = "jaxlib-${version}-cp312-cp312-${platform}.whl";
            # Base URL for jaxlib wheels on PyPI
            baseUrl = "https://files.pythonhosted.org/packages";
            # These are the specific paths for each platform's wheel
            wheelUrls = {
              "manylinux2014_x86_64" = "${baseUrl}/b1/09/58d35465d48c8bee1d9a4e7a3c5db2edaabfc7ac94f4576c9f8c51b83e70/${wheelName}";
              "manylinux2014_aarch64" = "${baseUrl}/e0/af/10b49f8de2acc7abc871478823579d7241be52ca0d6bb0d2b2c476cc1b68/${wheelName}";
              "macosx_11_0_arm64" = "${baseUrl}/68/cf/28895a4a89d88d18592507d7a35218b6bb2d8bced13615065c9f925f2ae1/${wheelName}";
            };
          in
            pkgs.fetchurl {
              url = wheelUrls.${platform} or (throw "Unsupported platform: ${platform}");
              hash =
                if system == "x86_64-linux"
                then "sha256-Hxr6X9WKYPZ/DKWG4mcUrs5i6qLIM0wk0OgoWvxKfM0="
                else if system == "aarch64-linux"
                then "sha256-TYZ6GgVlsxz9qrvsgeAwLGRhuyrEuSwEZwMo15WBmAM="
                else "sha256-aPzyiJWkqonYjRhVkgfXo1KLsi287BMWFgXJvJL2Kh4="; # macosx_11_0_arm64
            };
        });
        jaxlib-bin = self.jaxlib; # Make jaxlib-bin point to our overridden jaxlib

        # Override JAX to use version 0.4.31
        jax = super.jax.overridePythonAttrs (old: rec {
          version = "0.4.31";
          src = super.fetchPypi {
            inherit (old) pname;
            inherit version;
            hash = "sha256-/S1HBkOgBz2CJzfweI9xORZWr35izFsueZXuOQzqwoc=";
          };
          # Dependencies will automatically use the overridden jaxlib from this package set
          # Skip version check during build
          pythonImportsCheck = [];
          doCheck = false;
        });
      };
    };
  in
    ps'.buildPythonPackage {
      pname = wheelName;
      format = "wheel";
      version = version;
      src = "${wheel}/${wheelName}-${wheelVersion}-${wheelSuffix}.whl";
      doCheck = false;
      propagatedBuildInputs = with ps'; [
        jax
        jaxlib
        typing-extensions
        numpy
        polars
        pytest
        matplotlib
      ];
      buildInputs = [
        xla_ext
        pkgs.gfortran.cc.lib
      ];
      nativeBuildInputs = with pkgs; (
        lib.optionals stdenv.isLinux [
          autoPatchelfHook
          patchelf
        ]
        ++ lib.optionals stdenv.isDarwin [
          fixDarwinDylibNames
          darwin.cctools
        ]
      );
      pythonImportsCheck = [wheelName];
    };
  py = elodin pkgs.python3Packages;
in {
  inherit py clippy;
}
