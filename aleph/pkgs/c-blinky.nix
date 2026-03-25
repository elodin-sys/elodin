{
  pkgs,
  lib,
  ...
}: let
  pname = "c-blinky";
  workspaceToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = workspaceToml.workspace.package.version;
  src = lib.cleanSourceWith {
    src = ../../fsw/c-blinky;
    filter = path: type: let
      base = baseNameOf (toString path);
    in
      !(type == "directory" && base == "tools")
      && !(lib.elem base [
        "firmware.elf"
        "firmware.bin"
        "firmware.elf.bin"
      ]);
  };
in
  pkgs.stdenvNoCC.mkDerivation {
    inherit pname version src;

    nativeBuildInputs = [
      pkgs.gcc-arm-embedded
    ];

    strictDeps = true;
    dontConfigure = true;

    buildPhase = ''
      runHook preBuild

      arm-none-eabi-gcc \
        -mcpu=cortex-m7 \
        -mthumb \
        -mfloat-abi=hard \
        -mfpu=fpv5-d16 \
        -Os \
        -g \
        -Wall \
        -Wextra \
        -ffunction-sections \
        -fdata-sections \
        -T linker.ld \
        -Wl,--gc-sections \
        -Wl,--print-memory-usage \
        main.c \
        -o firmware.elf

      arm-none-eabi-objcopy -O binary firmware.elf firmware.bin
      arm-none-eabi-size firmware.elf

      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall

      mkdir -p "$out/share/doc/${pname}"
      cp firmware.elf "$out/firmware.elf"
      cp firmware.bin "$out/firmware.bin"
      cp README.md "$out/share/doc/${pname}/README.md"

      runHook postInstall
    '';

    meta = {
      description = "Aleph STM32H7 c-blinky firmware artifact";
      platforms = lib.platforms.linux;
    };
  }
