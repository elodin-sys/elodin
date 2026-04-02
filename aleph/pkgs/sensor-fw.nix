{
  pkgs,
  rustToolchain,
  lib,
  gpsBaudRate ? 9600,
  ...
}: let
  pname = "sensor-fw";
  workspaceToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = workspaceToml.workspace.package.version;
  target = "thumbv7em-none-eabihf";

  sensorFwSrc = lib.cleanSourceWith {
    name = "sensor-fw-src";
    src = ../../fsw/sensor-fw;
    filter = path: type: let
      base = baseNameOf (toString path);
    in
      !(type == "directory" && base == "target")
      && !(type == "directory" && base == "tools")
      && !(lib.elem base [
        "firmware.elf"
        "firmware.bin"
      ]);
  };

  blackboxSrc = lib.cleanSourceWith {
    name = "blackbox-src";
    src = ../../fsw/blackbox;
    filter = path: type: let
      base = baseNameOf (toString path);
    in
      (type == "directory")
      || (lib.any (suffix: lib.hasSuffix suffix base) [
        ".rs"
        ".toml"
      ]);
  };

  cargoVendorDir = pkgs.rustPlatform.importCargoLock {
    lockFile = ../../fsw/sensor-fw/Cargo.lock;
    allowBuiltinFetchGit = true;
  };

  toolchain = rustToolchain pkgs;
in
  pkgs.stdenvNoCC.mkDerivation {
    inherit pname version;

    srcs = [sensorFwSrc blackboxSrc];
    sourceRoot = ".";

    nativeBuildInputs = [
      toolchain
      pkgs.gcc-arm-embedded
    ];

    strictDeps = true;
    dontConfigure = true;
    dontFixup = true;

    buildPhase = ''
      runHook preBuild

      export HOME=$TMPDIR

      # Lay out sources so the path dep ../blackbox resolves
      mkdir -p build/fsw
      cp -r sensor-fw-src build/fsw/sensor-fw
      cp -r blackbox-src build/fsw/blackbox

      # Patch blackbox Cargo.toml: replace workspace-inherited fields
      sed -i \
        -e 's/version\.workspace = true/version = "${version}"/' \
        -e 's/edition\.workspace = true/edition = "2021"/' \
        -e 's/repository\.workspace = true/repository = "https:\/\/github.com\/elodin-sys\/elodin"/' \
        build/fsw/blackbox/Cargo.toml

      cd build/fsw/sensor-fw

      mkdir -p .cargo
      cat > .cargo/config.toml <<CARGO_CFG
      [target.thumbv7em-none-eabihf]
      rustflags = ["-C", "link-arg=-Tlink.x", "-C", "link-arg=-Tdefmt.x"]

      [source.crates-io]
      replace-with = "vendored-sources"

      [source."git+https://github.com/akhilles/stm32-hal.git?rev=a06e441"]
      git = "https://github.com/akhilles/stm32-hal.git"
      rev = "a06e441"
      replace-with = "vendored-sources"

      [source."git+https://github.com/rafalh/rust-fatfs.git?rev=c4bb769"]
      git = "https://github.com/rafalh/rust-fatfs.git"
      rev = "c4bb769"
      replace-with = "vendored-sources"

      [source.vendored-sources]
      directory = "${cargoVendorDir}"
      CARGO_CFG

      cargo build --target ${target} --release --bin fw --offline ${
        lib.optionalString (gpsBaudRate == 38400) "--features gps-38400"
      }

      arm-none-eabi-objcopy -O binary target/${target}/release/fw firmware.bin
      arm-none-eabi-size target/${target}/release/fw

      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall

      mkdir -p "$out/share/doc/${pname}"
      cp target/${target}/release/fw "$out/firmware.elf"
      cp firmware.bin "$out/firmware.bin"
      cp README.md "$out/share/doc/${pname}/README.md"

      runHook postInstall
    '';

    meta = {
      description = "Aleph STM32H7 sensor firmware (IMU/mag/baro streaming)";
      platforms = lib.platforms.linux;
    };
  }
