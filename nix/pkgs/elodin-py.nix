{
  pkgs,
  lib,
  rustToolchain,
  system,
  python,
  pythonPackages,
  ...
}: let
  # Direct Rust build using rustPlatform.buildRustPackage
  xla_ext = pkgs.callPackage ./xla-ext.nix {inherit system;};

  # Extract pname and version directly from Cargo.toml files
  noxPyCargoToml = builtins.fromTOML (builtins.readFile ../../libs/nox-py/Cargo.toml);
  workspaceCargoToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  pname = noxPyCargoToml.package.name;
  version = workspaceCargoToml.workspace.package.version;

  src = pkgs.nix-gitignore.gitignoreSource [] ../..;

  arch = with pkgs;
    if stdenv.isDarwin
    then
      # Python wheels require "arm64" for ARM Macs, not "aarch64"
      if stdenv.hostPlatform.ubootArch == "aarch64"
      then "arm64"
      else stdenv.hostPlatform.ubootArch
    else builtins.elemAt (lib.strings.splitString "-" system) 0;

  wheelName = "elodin";
  wheelPlatform =
    if pkgs.stdenv.isDarwin
    then "macosx_11_0"
    else "linux";
  wheelSuffix = "cp310-abi3-${wheelPlatform}_${arch}";
  # Convert version format from 0.15.0-alpha.1 to 0.15.0a1 for wheel filename
  wheelVersion = lib.strings.replaceStrings ["-alpha."] ["a"] version;

  # Build the wheel using rustPlatform
  wheel = pkgs.rustPlatform.buildRustPackage rec {
    inherit pname version src;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "libs/nox-py";

    nativeBuildInputs = with pkgs;
      [
        (rustToolchain pkgs)
        maturin
        pkg-config
        cmake
        python3 # Add python3 to nativeBuildInputs so it's available during build
        which # Required for build scripts that use which to find python3
        gfortran # Fortran compiler needed at build time for netlib-src
      ]
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
        python
        openssl
        gfortran.cc.lib # Fortran runtime library for linking
        xla_ext
      ]
      ++ lib.optionals stdenv.isDarwin [
        libiconv
      ];

    # Environment variables for the build
    XLA_EXTENSION_DIR = "${xla_ext}";
    OPENSSL_DIR = "${pkgs.openssl.dev}";
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
    OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include/";

    # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
    # GCC 14 treats -Wincompatible-pointer-types as error by default
    NIX_CFLAGS_COMPILE = lib.optionalString pkgs.stdenv.isLinux "-Wno-error=incompatible-pointer-types";

    doCheck = false;

    # Override the build phase to use maturin
    buildPhase = ''
      runHook preBuild

      # Build the wheel with maturin
      maturin build --offline --target-dir ./target -m libs/nox-py/Cargo.toml --release

      runHook postBuild
    '';

    # Install the wheel
    installPhase = ''
      runHook preInstall

      mkdir -p $out
      cp target/wheels/${wheelName}-${wheelVersion}-${wheelSuffix}.whl $out/

      runHook postInstall
    '';
  };

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
  py = elodin pythonPackages;
in {
  # Note: clippy is not included here since we're bypassing crane
  # Run clippy directly via cargo in the development shell (nix develop)
  inherit py python pythonPackages;
}
