{
  pkgs,
  lib,
  rustToolchain,
  python,
  pythonPackages,
  ...
}: let
  common = pkgs.callPackage ./common.nix {};

  noxPyCargoToml = builtins.fromTOML (builtins.readFile ../../libs/nox-py/Cargo.toml);
  workspaceCargoToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  pname = noxPyCargoToml.package.name;
  version = workspaceCargoToml.workspace.package.version;

  src = pkgs.nix-gitignore.gitignoreSource [] ../..;

  arch = with pkgs;
    if stdenv.isDarwin
    then
      if stdenv.hostPlatform.ubootArch == "aarch64"
      then "arm64"
      else stdenv.hostPlatform.ubootArch
    else builtins.elemAt (lib.strings.splitString "-" stdenv.hostPlatform.system) 0;

  wheelName = "elodin";
  wheelPlatform =
    if pkgs.stdenv.isDarwin
    then "macosx_11_0"
    else "linux";
  wheelSuffix = "cp310-abi3-${wheelPlatform}_${arch}";
  wheelVersion = lib.strings.replaceStrings ["-alpha."] ["a"] version;

  maturinFeatures = "--features publish";

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
        python3
        which
      ]
      ++ common.commonNativeBuildInputs
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
      ]
      ++ common.commonBuildInputs
      ++ lib.optionals stdenv.isDarwin common.darwinDeps
      ++ lib.optionals stdenv.isDarwin [
        stdenv.cc.cc.lib
      ];

    OPENSSL_DIR = "${pkgs.openssl.dev}";
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
    OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include/";

    NIX_LDFLAGS = lib.optionalString pkgs.stdenv.isDarwin "-lc++ -headerpad_max_install_names";

    doCheck = false;

    buildPhase = ''
      runHook preBuild

      maturin build --offline --target-dir ./target -m libs/nox-py/Cargo.toml --release ${maturinFeatures}

      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall

      mkdir -p $out
      cp target/wheels/*.whl "$out/${wheelName}-${wheelVersion}-${wheelSuffix}.whl"

      runHook postInstall
    '';
  };

  elodin = ps:
    ps.buildPythonPackage {
      pname = wheelName;
      format = "wheel";
      version = version;
      src = "${wheel}/${wheelName}-${wheelVersion}-${wheelSuffix}.whl";
      doCheck = false;
      pythonImportsCheck = [];
      propagatedBuildInputs = with ps;
        [
          jax
          jaxlib
          typing-extensions
          numpy
          polars
          pytest
          matplotlib
        ]
        ++ lib.optionals pkgs.stdenv.isDarwin [
          pkgs.libcxx
        ];
      buildInputs = lib.optionals pkgs.stdenv.isDarwin [
        pkgs.stdenv.cc.cc.lib
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
    };
  py = elodin pythonPackages;
in {
  inherit py python pythonPackages;
}
