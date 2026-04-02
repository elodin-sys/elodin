{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.serial-bridge;
in {
  options.services.serial-bridge = {
    enable = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Whether to run the Aleph serial bridge service.";
    };

    serialPort = lib.mkOption {
      type = lib.types.str;
      default = "/dev/ttyTHS1";
      description = "Serial device used for the Aleph STM32 board-to-board link.";
    };

    baudRate = lib.mkOption {
      type = lib.types.int;
      default = 115200;
      description = "Baud rate used when opening the Aleph STM32 board-to-board serial device.";
    };

    gpsClockSource = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Use GPS-derived timestamps for all sensor data. When enabled, no records are written until GPS time is valid.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.serial-bridge = with pkgs; {
      wantedBy = ["multi-user.target"];
      after = ["network.target"];
      description = "start aleph-serial-bridge";
      serviceConfig = {
        Type = "exec";
        User = "root";
        ExecStart = "${serial-bridge}/bin/aleph-serial-bridge";
        KillSignal = "SIGINT";
        Environment = [
          "RUST_LOG=debug"
          "ALEPH_SERIAL_BRIDGE_PORT=${cfg.serialPort}"
          "ALEPH_SERIAL_BRIDGE_BAUD=${toString cfg.baudRate}"
          "ALEPH_SERIAL_BRIDGE_GPS_CLOCK=${lib.boolToString cfg.gpsClockSource}"
        ];
      };
    };
  };
}
