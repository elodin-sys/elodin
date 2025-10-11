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

  # Import shared JAX overrides
  jaxOverrides = pkgs.callPackage ./jax-overrides.nix {inherit pkgs;};
  
  elodin = ps: let
    # Create a modified Python package set with our JAX/jaxlib overrides
    # This ensures all packages use the same jaxlib version
    ps' = ps.override {
      overrides = jaxOverrides;
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
