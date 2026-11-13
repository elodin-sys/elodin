{
  pkgs,
  lib,
  rustToolchain,
  ...
}: let
  pname = "elodin-web";
  cargoToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = cargoToml.workspace.package.version;
  root = toString ../..;
  common = import ./src.nix {inherit lib;};
  # The editor's sources plus its wasm rustflags and Trunk page.
  webFiles = ["/.cargo" "/.cargo/config.toml" "/apps/elodin-web/index.html" "/apps/elodin-web/Trunk.toml"];
  src = lib.cleanSourceWith {
    src = ../..;
    filter = path: type: common.filter path type || lib.elem (lib.removePrefix root path) webFiles;
  };

  # Trunk runs the wasm-bindgen CLI matching Cargo.lock, which is newer than nixpkgs'.
  bindgenVersion = "0.2.129";
  lockedBindgenVersion =
    (lib.findFirst (package: package.name == "wasm-bindgen") null
      (builtins.fromTOML (builtins.readFile ../../Cargo.lock)).package).version;
  bindgen = assert lib.assertMsg (bindgenVersion == lockedBindgenVersion)
  "nix/pkgs/elodin-web.nix: set bindgenVersion and its hashes to wasm-bindgen ${lockedBindgenVersion} from Cargo.lock";
    pkgs.buildWasmBindgenCli rec {
      src = pkgs.fetchCrate {
        pname = "wasm-bindgen-cli";
        version = bindgenVersion;
        hash = "sha256-pcecKQd7E8Opw6bkFoE569epUi7gh5qpQF1e5PJY6V8=";
      };
      cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
        inherit src;
        inherit (src) pname version;
        hash = "sha256-vmUrWVU7kPJJxO5qIVeAkwQyWDELO1Z4Z5gitz2kco8=";
      };
    };

  toolchain = rustToolchain pkgs;
  rustPlatform = pkgs.makeRustPlatform {
    cargo = toolchain;
    rustc = toolchain;
  };
in
  # The browser editor (`just web-release`): Trunk's dist for the wasm-release profile, served from `/`.
  # Hosts that serve it elsewhere load `elodin-web-*.js` and `elodin-web-*_bg.wasm` with their own page.
  rustPlatform.buildRustPackage {
    inherit pname version src;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    # cargo-auditable's linker section breaks the wasm32 link.
    auditable = false;

    nativeBuildInputs = [
      pkgs.trunk
      bindgen
      pkgs.pkg-config
      pkgs.cmake
      pkgs.python3
    ];

    # C build scripts (zstd-sys) target wasm32 with the unwrapped compiler and their own wasm headers.
    CC_wasm32_unknown_unknown = "${pkgs.llvmPackages.clang-unwrapped}/bin/clang";

    doCheck = false;

    buildPhase = ''
      runHook preBuild

      # A checkout without Git LFS holds pointer files; reject them before embedding.
      python3 - <<'PY'
      from pathlib import Path
      for path in Path('libs/elodin-editor').rglob('*'):
          if path.is_file():
              with path.open('rb') as source:
                  if source.read(42).startswith(b'version https://git-lfs.github.com/spec/'):
                      raise SystemExit(f'Unresolved Git LFS asset: {path}')
      PY

      (cd apps/elodin-web && trunk build --offline --locked --cargo-profile wasm-release)

      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      mkdir -p "$out"
      cp -r apps/elodin-web/dist/. "$out/"
      runHook postInstall
    '';

    meta = {
      description = "Elodin browser editor (WASM)";
      platforms = lib.platforms.all;
    };
  }
