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

    [osd]
    rows = ${toString cfg.osdRows}
    cols = ${toString cfg.osdCols}
    refresh_rate_hz = ${toString cfg.refreshRateHz}
    coordinate_frame = "${cfg.coordinateFrame}"

    [serial]
    port = "${cfg.serialPort}"
    baud = ${toString cfg.baudRate}

    # Input mappings for extracting telemetry from Elodin-DB components
    [inputs.position]
    component = "${cfg.inputs.position.component}"
    x = ${toString cfg.inputs.position.x}
    y = ${toString cfg.inputs.position.y}
    z = ${toString cfg.inputs.position.z}

    [inputs.orientation]
    component = "${cfg.inputs.orientation.component}"
    qx = ${toString cfg.inputs.orientation.qx}
    qy = ${toString cfg.inputs.orientation.qy}
    qz = ${toString cfg.inputs.orientation.qz}
    qw = ${toString cfg.inputs.orientation.qw}

    [inputs.velocity]
    component = "${cfg.inputs.velocity.component}"
    x = ${toString cfg.inputs.velocity.x}
    y = ${toString cfg.inputs.velocity.y}
    z = ${toString cfg.inputs.velocity.z}
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

    # Input mappings for extracting telemetry from Elodin-DB components
    inputs = {
      position = {
        component = mkOption {
          type = types.str;
          default = "bdx.world_pos";
          description = "Component name for position data (e.g., 'bdx.world_pos')";
        };
        x = mkOption {
          type = types.int;
          default = 4;
          description = "Array index for X position";
        };
        y = mkOption {
          type = types.int;
          default = 5;
          description = "Array index for Y position";
        };
        z = mkOption {
          type = types.int;
          default = 6;
          description = "Array index for Z position (altitude)";
        };
      };

      orientation = {
        component = mkOption {
          type = types.str;
          default = "bdx.world_pos";
          description = "Component name for orientation quaternion data";
        };
        qx = mkOption {
          type = types.int;
          default = 0;
          description = "Array index for quaternion X component";
        };
        qy = mkOption {
          type = types.int;
          default = 1;
          description = "Array index for quaternion Y component";
        };
        qz = mkOption {
          type = types.int;
          default = 2;
          description = "Array index for quaternion Z component";
        };
        qw = mkOption {
          type = types.int;
          default = 3;
          description = "Array index for quaternion W (scalar) component";
        };
      };

      velocity = {
        component = mkOption {
          type = types.str;
          default = "bdx.world_vel";
          description = "Component name for velocity data (e.g., 'bdx.world_vel')";
        };
        x = mkOption {
          type = types.int;
          default = 3;
          description = "Array index for X velocity";
        };
        y = mkOption {
          type = types.int;
          default = 4;
          description = "Array index for Y velocity";
        };
        z = mkOption {
          type = types.int;
          default = 5;
          description = "Array index for Z velocity";
        };
      };
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
