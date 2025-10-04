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

  pyFilter = path: _type: builtins.match ".*py$" path != null;
  mdFilter = path: _type: builtins.match ".*md$" path != null;
  protoFilter = path: _type: builtins.match ".*proto$" path != null;
  assetFilter = path: _type: builtins.match ".*assets.*$" path != null;
  cppFilter = path: _type: builtins.match ".*[h|(cpp)|(cpp.jinja)]$" path != null;

  srcFilter = path: type:
    (pyFilter path type)
    || (mdFilter path type)
    || (protoFilter path type)
    || (assetFilter path type)
    || (cppFilter path type)
    || (craneLib.filterCargoSources path type);

  src = lib.cleanSourceWith {
    src = craneLib.path ./../..;
    filter = srcFilter;
  };

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

  elodin = ps:
    ps.buildPythonPackage {
      pname = wheelName;
      format = "wheel";
      version = version;
      src = "${wheel}/${wheelName}-${wheelVersion}-${wheelSuffix}.whl";
      doCheck = false;
      propagatedBuildInputs = with ps; [
        jax
        jaxlib
        typing-extensions
        numpy
        polars
        pytest
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
