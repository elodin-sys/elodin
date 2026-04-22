{
  config,
  lib,
  pkgs,
  ...
}:
with lib; let
  cfg = config.services.sensor-fw;

  flash = import ../lib/mk-mcu-flash-service.nix {
    inherit pkgs lib cfg;
    name = "sensor-fw";
    timeoutSec = "300s";
  };
in {
  options.services.sensor-fw = {
    enable = mkOption {
      type = types.bool;
      default = false;
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

    gps.model = mkOption {
      type = types.nullOr (types.enum ["m10q" "m9n"]);
      default = null;
      description = ''
        u-blox GPS module connected on the J7 connector.
        Set to "m10q" for SAM-M10Q or "m9n" for NEO-M9N / M9N-5883.
        When set, GPS-disciplined timestamping is automatically enabled.
        When null (default), no GPS is expected and timestamping uses wall-clock.
      '';
    };
  };

  config = mkIf cfg.enable {
    services.sensor-fw.package = mkDefault (pkgs.sensor-fw.override {
      gpsBaudRate =
        if cfg.gps.model == "m9n"
        then 38400
        else 9600;
    });

    services.serial-bridge.gpsClockSource = mkDefault (cfg.gps.model != null);

    environment.systemPackages = [cfg.package];

    services.serial-bridge.baudRate = 1000000;

    systemd.services.sensor-fw-flash = flash.serviceAttrs;

    systemd.services.serial-bridge = {
      after = ["sensor-fw-flash.service"];
    };
  };
}
