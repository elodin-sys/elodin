{
  pkgs,
  lib,
  rustToolchain,
  system,
  python,
  pythonPackages,
  enableTracy ? false,
  iree_runtime ?
    pkgs.callPackage ./iree-runtime.nix {
      enableCuda = pkgs.stdenv.isLinux;
      enableMetal = pkgs.stdenv.isDarwin;
    },
  iree_runtime_tracy ? null,
  ...
}: let
  # Import shared configuration
  common = pkgs.callPackage ./common.nix {};
  iree_compiler = pkgs.callPackage ./iree-compiler.nix {python3 = python;};
  iree_compiler_source = pkgs.callPackage ./iree-compiler-source.nix {};
  iree_runtime_used =
    if enableTracy
    then assert iree_runtime_tracy != null; iree_runtime_tracy
    else iree_runtime;
  # Direct Rust build using rustPlatform.buildRustPackage

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
        python3 # Add python3 to nativeBuildInputs so it's available during build
        which # Required for build scripts that use which to find python3
        llvmPackages.libclang # Required for bindgen (used by iree-runtime)
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
        iree_runtime_used
      ]
      ++ common.commonBuildInputs
      ++ lib.optionals stdenv.isDarwin common.darwinDeps
      ++ lib.optionals stdenv.isDarwin [
        stdenv.cc.cc.lib # C++ standard library
      ];

    # Environment variables for the build
    IREE_RUNTIME_DIR = "${iree_runtime}";
    IREE_RUNTIME_TRACY_DIR = lib.optionalString enableTracy "${iree_runtime_used}";
    OPENBLAS_DIR = "${pkgs.openblas}";
    OPENSSL_DIR = "${pkgs.openssl.dev}";
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
    OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include/";
    LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

    # Tell bindgen where to find C standard library headers (required on Linux in Nix builds)
    BINDGEN_EXTRA_CLANG_ARGS =
      lib.optionalString pkgs.stdenv.isLinux
      "-I${pkgs.stdenv.cc.libc.dev}/include -I${pkgs.llvmPackages.libclang.lib}/lib/clang/${lib.versions.major pkgs.llvmPackages.libclang.version}/include";

    # Ensure C++ standard library is linked on macOS
    NIX_LDFLAGS = lib.optionalString pkgs.stdenv.isDarwin "-lc++";

    doCheck = false;

    # Override the build phase to use maturin
    buildPhase = ''
      runHook preBuild

      # Build the wheel with maturin
      maturin build --offline --target-dir ./target -m libs/nox-py/Cargo.toml --release ${lib.optionalString enableTracy "--features publish,tracy"}

      runHook postBuild
    '';

    # Install the wheel (maturin's output name can vary by platform; copy to fixed name for downstream)
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
      pythonImportsCheck = []; # Skip import check due to C++ library loading issues on macOS
      propagatedBuildInputs = with ps;
        [
          jax
          jaxlib
          typing-extensions
          numpy
          polars
          pytest
          matplotlib
          iree_compiler # IREE compiler Python package for iree.compiler imports
          iree_compiler_source # Source-built iree-compile binary (patchable)
        ]
        ++ lib.optionals pkgs.stdenv.isDarwin [
          pkgs.libcxx # C++ standard library runtime
        ];
      buildInputs =
        [
          iree_runtime_used
        ]
        ++ lib.optionals pkgs.stdenv.isDarwin [
          pkgs.stdenv.cc.cc.lib # C++ standard library for macOS
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
      postFixup = ''
        for d in $out/lib/python*/site-packages/elodin; do
          echo "${iree_compiler_source}" > "$d/_iree_compiler_dir"
        done
      '';
    };
  py = elodin pythonPackages;
in {
  inherit py python pythonPackages;
}
