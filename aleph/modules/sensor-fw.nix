{
  config,
  lib,
  pkgs,
  ...
}:
with lib; let
  cfg = config.services.sensor-fw;

  resetMcu = pkgs.writeShellApplication {
    name = "reset-mcu";
    runtimeInputs = [
      pkgs.coreutils
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

  sensorFwFlash = pkgs.writeShellApplication {
    name = "sensor-fw-flash-service";
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
        echo "sensor-fw firmware already flashed from $desired_firmware"
        exit 0
      fi

      echo "Flashing sensor-fw firmware from $desired_firmware"
      flash-mcu --bin "$desired_firmware"
      printf '%s\n' "$desired_signature" > "$marker"
    '';
  };
in {
  options.services.sensor-fw = {
    enable = mkOption {
      type = types.bool;
      default = true;
      description = "Whether to deploy and flash sensor-fw onto the Aleph STM32H7.";
    };

    autostart = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Whether to auto-run the sensor-fw flash service.
        When false, the unit is configured but must be started manually.
      '';
    };

    package = mkOption {
      type = types.package;
      default = pkgs.sensor-fw;
      description = "Firmware package containing the sensor-fw ELF and BIN artifacts.";
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
    assertions = optionals (builtins.hasAttr "c-blinky" config.services) [
      {
        assertion = !config.services.c-blinky.enable;
        message = "services.sensor-fw and services.c-blinky are mutually exclusive (both flash the STM32).";
      }
    ];

    environment.systemPackages = [cfg.package];

    services.serial-bridge.baudRate = 1000000;

    systemd.services.sensor-fw-flash = {
      description = "Flash sensor-fw onto the Aleph STM32H7";
      wantedBy = mkIf cfg.autostart ["multi-user.target"];
      before = ["serial-bridge.service"];
      restartIfChanged = true;
      restartTriggers = [cfg.package];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "root";
        Group = "root";
        StateDirectory = "sensor-fw-flash";
        ExecStart = "${sensorFwFlash}/bin/sensor-fw-flash-service";
        TimeoutStartSec = "300s";
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
      after = ["sensor-fw-flash.service"];
    };
  };
}
