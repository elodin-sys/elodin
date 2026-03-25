{
  config,
  lib,
  pkgs,
  ...
}:
with lib; let
  cfg = config.services.c-blinky;

  resetMcu = pkgs.writeShellApplication {
    name = "reset-mcu";
    runtimeInputs = [
      pkgs.coreutils
      pkgs.i2c-tools
      pkgs.libgpiod_1
      pkgs.procps
    ];
    text = builtins.readFile ../scripts/reset-mcu.sh;
  };

  flashMcu = pkgs.writeShellApplication {
    name = "flash-mcu";
    runtimeInputs = [
      pkgs.coreutils
      pkgs.gcc-arm-embedded
      pkgs.lsof
      pkgs.openocd
      pkgs.procps
      pkgs.stm32flash
      pkgs.systemd
      resetMcu
    ];
    text = builtins.readFile ../scripts/flash-mcu.sh;
  };

  cBlinkyFlash = pkgs.writeShellApplication {
    name = "c-blinky-flash-service";
    runtimeInputs = [
      pkgs.coreutils
      flashMcu
    ];
    text = ''
      marker="$STATE_DIRECTORY/last-flashed-package"
      desired_signature="${concatStringsSep "|" [
        (toString cfg.package)
        (toString flashMcu)
        cfg.serialPort
        (toString cfg.bootloaderBaudRate)
        cfg.boot0GpioChip
        (toString cfg.boot0GpioLine)
        cfg.resetGpioChip
        (toString cfg.resetGpioLine)
      ]}"
      desired_firmware="${cfg.package}/firmware.bin"

      if [ -f "$marker" ] && [ "$(cat "$marker")" = "$desired_signature" ]; then
        echo "c-blinky firmware already flashed from $desired_firmware"
        exit 0
      fi

      echo "Flashing c-blinky firmware from $desired_firmware"
      flash-mcu --bin "$desired_firmware"
      printf '%s\n' "$desired_signature" > "$marker"
    '';
  };
in {
  options.services.c-blinky = {
    enable = mkOption {
      type = types.bool;
      default = true;
      description = "Whether to deploy and flash c-blinky onto the Aleph STM32H7.";
    };

    autostart = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Whether to auto-run the c-blinky flash service.
        When false, the unit is configured but must be started manually.
      '';
    };

    package = mkOption {
      type = types.package;
      default = pkgs.c-blinky;
      description = "Firmware package containing the c-blinky ELF and BIN artifacts.";
    };

    serialPort = mkOption {
      type = types.str;
      default = config.services.serial-bridge.serialPort;
      description = "UART device used to reach the STM32 ROM bootloader.";
    };

    bootloaderBaudRate = mkOption {
      type = types.int;
      default = 19200;
      description = "Baud rate used when probing and flashing the STM32 ROM bootloader.";
    };

    boot0GpioChip = mkOption {
      type = types.str;
      default = "gpiochip0";
      description = "GPIO chip used to hold STM32 BOOT0 high on the open-source expansion board.";
    };

    boot0GpioLine = mkOption {
      type = types.int;
      default = 144;
      description = "GPIO line used to hold STM32 BOOT0 high on the open-source expansion board.";
    };

    resetGpioChip = mkOption {
      type = types.str;
      default = "gpiochip0";
      description = "GPIO chip used to pulse STM32 NRST on the open-source expansion board.";
    };

    resetGpioLine = mkOption {
      type = types.int;
      default = 106;
      description = "GPIO line used to pulse STM32 NRST on the open-source expansion board.";
    };
  };

  config = mkIf cfg.enable {
    environment.systemPackages = [cfg.package];

    systemd.services.c-blinky-flash = {
      description = "Flash c-blinky onto the Aleph STM32H7";
      wantedBy = mkIf cfg.autostart ["multi-user.target"];
      before = ["serial-bridge.service"];
      restartIfChanged = true;
      restartTriggers = [cfg.package];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "root";
        Group = "root";
        StateDirectory = "c-blinky-flash";
        ExecStart = "${cBlinkyFlash}/bin/c-blinky-flash-service";
        TimeoutStartSec = "90s";
        Environment = [
          "ALEPH_FLASH_MCU_METHOD=uart"
          "ALEPH_FLASH_MCU_ADDR=0x08000000"
          "ALEPH_FLASH_MCU_PORT=${cfg.serialPort}"
          "ALEPH_FLASH_MCU_BAUD=${toString cfg.bootloaderBaudRate}"
          "ALEPH_FLASH_MCU_BRIDGE_UNIT=serial-bridge.service"
          "ALEPH_BOOT0_GPIOCHIP=${cfg.boot0GpioChip}"
          "ALEPH_BOOT0_GPIOLINE=${toString cfg.boot0GpioLine}"
          "ALEPH_NRST_GPIOCHIP=${cfg.resetGpioChip}"
          "ALEPH_NRST_GPIOLINE=${toString cfg.resetGpioLine}"
        ];
        StandardOutput = "journal";
        StandardError = "journal";
      };
    };

    systemd.services.serial-bridge = {
      after = ["c-blinky-flash.service"];
    };
  };
}
