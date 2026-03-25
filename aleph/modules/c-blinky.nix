{
  config,
  lib,
  pkgs,
  ...
}:
with lib; let
  cfg = config.services.c-blinky;

  flash = import ../lib/mk-mcu-flash-service.nix {
    inherit pkgs lib cfg;
    name = "c-blinky";
    timeoutSec = "90s";
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

    systemd.services.c-blinky-flash = flash.serviceAttrs;

    systemd.services.serial-bridge = {
      after = ["c-blinky-flash.service"];
    };
  };
}
