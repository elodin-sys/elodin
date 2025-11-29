{
  config,
  lib,
  pkgs,
  ...
}:
with lib; let
  cfg = config.services.msp-osd;
  configFile = pkgs.writeText "config.toml" ''
    [db]
    host = "${cfg.dbHost}"
    port = ${toString cfg.dbPort}
    components = [
      ${concatStringsSep ",\n      " (map (c: ''"${c}"'') cfg.components)}
    ]

    [osd]
    rows = ${toString cfg.osdRows}
    cols = ${toString cfg.osdCols}
    refresh_rate_hz = ${toString cfg.refreshRateHz}
    coordinate_frame = "${cfg.coordinateFrame}"

    [serial]
    port = "${cfg.serialPort}"
    baud = ${toString cfg.baudRate}
  '';
in {
  options.services.msp-osd = {
    enable = mkEnableOption "MSP OSD service for MSP DisplayPort";

    package = mkOption {
      type = types.package;
      default = pkgs.msp-osd;
      description = "MSP OSD package to use";
    };

    mode = mkOption {
      type = types.enum ["serial" "debug"];
      default = "serial";
      description = "Operation mode: serial for MSP DisplayPort, debug for terminal";
    };

    dbHost = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = "Elodin-DB host address";
    };

    dbPort = mkOption {
      type = types.port;
      default = 2240;
      description = "Elodin-DB port";
    };

    components = mkOption {
      type = types.listOf types.str;
      default = [
        "gyro"
        "accel"
        "magnetometer"
        "attitude_target"
        "body_ang_vel"
        "world_pos"
        "world_vel"
      ];
      description = "List of components to subscribe to from Elodin-DB";
    };

    osdRows = mkOption {
      type = types.int;
      default = 18;
      description = "OSD grid rows";
    };

    osdCols = mkOption {
      type = types.int;
      default = 50;
      description = "OSD grid columns";
    };

    refreshRateHz = mkOption {
      type = types.float;
      default = 20.0;
      description = "OSD refresh rate in Hz";
    };

    coordinateFrame = mkOption {
      type = types.enum ["enu" "ned"];
      default = "ned";
      description = ''
        Coordinate frame convention for heading interpretation:
        - "enu" (East-North-Up): 0°=East, 90°=North (Elodin simulation default)
        - "ned" (North-East-Down): 0°=North, 90°=East (Aviation convention)

        Defaults to "ned" for real aviation hardware. Use "enu" when testing
        with Elodin simulations.
      '';
    };

    serialPort = mkOption {
      type = types.str;
      default = "/dev/ttyTHS7";
      description = "Serial port for MSP DisplayPort communication";
    };

    baudRate = mkOption {
      type = types.int;
      default = 115200;
      description = "Serial baud rate";
    };

    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Extra command-line arguments to pass to msp-osd";
    };
  };

  config = mkIf cfg.enable {
    systemd.services.msp-osd = {
      description = "MSP DisplayPort OSD Service";
      wantedBy = ["multi-user.target"];
      after = ["network.target" "elodin-db.service"];
      wants = ["elodin-db.service"];

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/msp-osd --config ${configFile} --mode ${cfg.mode} ${concatStringsSep " " cfg.extraArgs}";
        Restart = "always";
        RestartSec = 5;
        StandardOutput = "journal";
        StandardError = "journal";

        # Run as root for serial port access (or configure udev rules)
        User = "root";

        # Device access for serial port
        DeviceAllow = [
          "${cfg.serialPort} rw"
        ];
        PrivateDevices = false;

        # Security hardening
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
      };

      environment = {
        RUST_LOG = mkDefault "info";
      };
    };

    # Ensure the serial port is accessible
    services.udev.extraRules = mkIf (cfg.mode == "serial") ''
      KERNEL=="ttyTHS[0-9]*", MODE="0666", GROUP="dialout"
    '';
  };
}
