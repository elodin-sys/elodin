{
  pkgs,
  rustToolchain,
  lib,
  elodinPy,
  python,
  pythonPackages,
  ...
}: let
  # Import shared configuration
  common = pkgs.callPackage ./common.nix {};
  pname = "elodin";
  # Derive version from Cargo.toml workspace configuration
  cargoToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = cargoToml.workspace.package.version;
  src = pkgs.nix-gitignore.gitignoreSource [] ../..;
  pythonPath = pythonPackages.makePythonPath [elodinPy];
  pythonMajorMinor = lib.versions.majorMinor python.version;

  bin = pkgs.rustPlatform.buildRustPackage rec {
    inherit pname version src;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "apps/elodin";

    nativeBuildInputs = with pkgs;
      [
        (rustToolchain pkgs)
        makeWrapper # Required for wrapProgram in postInstall
      ]
      ++ common.commonNativeBuildInputs;

    buildInputs = with pkgs;
      [
        python
      ]
      ++ common.commonBuildInputs
      ++ lib.optionals pkgs.stdenv.isDarwin common.darwinDeps
      ++ lib.optionals pkgs.stdenv.isLinux common.linuxGraphicsAudioDeps;

    doCheck = false;

    postInstall = ''
      wrapProgram $out/bin/elodin \
        ${common.makeWrapperArgs {
        inherit pkgs python pythonPath pythonMajorMinor;
      }}
    '';

    # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
    # GCC 14 treats -Wincompatible-pointer-types as error by default
    NIX_CFLAGS_COMPILE = common.netlibWorkaround;
    CARGO_PROFILE = "dev";
    CARGO_PROFILE_RELEASE_DEBUG = true;
  };
in
  bin
