{
  config,
  lib,
  pkgs,
  ...
}:
with lib; let
  cfg = config.services.msp-osd;
  configPath = "/etc/msp-osd/config.toml";
  configContents =
    ''
      [db]
      host = "${cfg.dbHost}"
      port = ${toString cfg.dbPort}

      [osd]
      rows = ${toString cfg.osdRows}
      cols = ${toString cfg.osdCols}
      refresh_rate_hz = ${toString cfg.refreshRateHz}
      coordinate_frame = "${cfg.coordinateFrame}"
      char_aspect_ratio = ${toString cfg.charAspectRatio}
      pitch_scale = ${toString cfg.pitchScale}

      [serial]
      port = "${cfg.serialPort}"
      baud = ${toString cfg.baudRate}
      auto_record = ${boolToString cfg.autoRecord}

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
    ''
    + lib.optionalString (cfg.inputs.target != null) ''

      # Target position for OSD target tracking indicator
      [inputs.target]
      component = "${cfg.inputs.target.component}"
      x = ${toString cfg.inputs.target.x}
      y = ${toString cfg.inputs.target.y}
      z = ${toString cfg.inputs.target.z}
    '';

  # Wrapper script to run msp-osd in debug mode with the deployed config
  mspOsdDebug = pkgs.writeShellScriptBin "msp-osd-debug" ''
    echo "Running msp-osd in debug mode..."
    echo "Config: ${configPath}"
    echo "Press Ctrl+C to exit"
    echo ""
    exec ${cfg.package}/bin/msp-osd --config ${configPath} --mode debug "$@"
  '';
in {
  options.services.msp-osd = {
    enable = mkEnableOption "MSP OSD service for MSP DisplayPort";

    autostart = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Whether to auto-start the MSP OSD service.
        When false, the service is configured but not started automatically.
        Use `systemctl start msp-osd` to start manually.
      '';
    };

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

    charAspectRatio = mkOption {
      type = types.float;
      default = 1.5;
      description = ''
        Character aspect ratio (height/width) for horizon line rendering.
        HD OSD systems like Walksnail Avatar use ~12x18 pixel characters (ratio 1.5).
        This compensates for non-square characters so the horizon tilt angle
        matches the actual aircraft roll angle.

        Common values:
        - 1.5: Walksnail Avatar, DJI HD (default)
        - 2.0: Standard SD analog OSD
        - 1.0: Square characters (no compensation)
      '';
    };

    pitchScale = mkOption {
      type = types.float;
      default = 5.0;
      description = ''
        Pitch scale in degrees per row for the artificial horizon.
        Lower values = more sensitive pitch response (horizon moves more per degree).
        Should be calibrated to match camera vertical FOV for accurate overlay.

        Formula: pitch_scale ≈ camera_vertical_fov / osd_rows
        Example: 90° VFOV / 18 rows ≈ 5° per row

        Common values:
        - 5.0: ~90° VFOV camera (default, good for Walksnail Avatar)
        - 6.0: ~108° VFOV camera
        - 4.0: ~72° VFOV camera (narrower FOV)
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

    autoRecord = mkOption {
      type = types.bool;
      default = false;
      description = ''
        Automatically start DVR recording on the VTX when msp-osd starts.
        Uses MSP2_COMMON_SET_RECORDING command (Walksnail Avatar compatible).
        Only applies when running in serial mode.
      '';
    };

    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Extra command-line arguments to pass to msp-osd";
    };

    # Input mappings for extracting telemetry from Elodin-DB components
    # Default mappings are configured for MEKF output (aleph.Output)
    inputs = {
      position = {
        component = mkOption {
          type = types.str;
          default = "aleph.Output";
          description = "Component name for position data (default: aleph.Output from MEKF)";
        };
        x = mkOption {
          type = types.int;
          default = 14;
          description = "Array index for X position (world_pos.linear[0] in MEKF output)";
        };
        y = mkOption {
          type = types.int;
          default = 15;
          description = "Array index for Y position (world_pos.linear[1] in MEKF output)";
        };
        z = mkOption {
          type = types.int;
          default = 16;
          description = "Array index for Z position/altitude (world_pos.linear[2] in MEKF output)";
        };
      };

      orientation = {
        component = mkOption {
          type = types.str;
          default = "aleph.Output";
          description = "Component name for orientation quaternion data (default: aleph.Output from MEKF)";
        };
        qx = mkOption {
          type = types.int;
          default = 0;
          description = "Array index for quaternion X component (q_hat[0] in MEKF output)";
        };
        qy = mkOption {
          type = types.int;
          default = 1;
          description = "Array index for quaternion Y component (q_hat[1] in MEKF output)";
        };
        qz = mkOption {
          type = types.int;
          default = 2;
          description = "Array index for quaternion Z component (q_hat[2] in MEKF output)";
        };
        qw = mkOption {
          type = types.int;
          default = 3;
          description = "Array index for quaternion W (scalar) component (q_hat[3] in MEKF output)";
        };
      };

      velocity = {
        component = mkOption {
          type = types.str;
          default = "aleph.Output";
          description = "Component name for velocity data (default: aleph.Output, placeholder until GPS integration)";
        };
        x = mkOption {
          type = types.int;
          default = 7;
          description = "Array index for X velocity (gyro_est[0] placeholder in MEKF output)";
        };
        y = mkOption {
          type = types.int;
          default = 8;
          description = "Array index for Y velocity (gyro_est[1] placeholder in MEKF output)";
        };
        z = mkOption {
          type = types.int;
          default = 9;
          description = "Array index for Z velocity (gyro_est[2] placeholder in MEKF output)";
        };
      };

      target = mkOption {
        type = types.nullOr (types.submodule {
          options = {
            component = mkOption {
              type = types.str;
              description = "Component name for target position data";
            };
            x = mkOption {
              type = types.int;
              description = "Array index for target X position";
            };
            y = mkOption {
              type = types.int;
              description = "Array index for target Y position";
            };
            z = mkOption {
              type = types.int;
              description = "Array index for target Z position";
            };
          };
        });
        default = null;
        description = ''
          Optional target position for OSD target tracking indicator.
          When set, displays direction and distance to target on the OSD.
          Used for tracking another aircraft or a waypoint.
        '';
      };
    };
  };

  config = mkIf cfg.enable {
    # Deploy config to /etc for easy access during debugging
    environment.etc."msp-osd/config.toml".text = configContents;

    # Add debug wrapper script to system packages
    environment.systemPackages = [cfg.package mspOsdDebug];

    systemd.services.msp-osd = {
      description = "MSP DisplayPort OSD Service";
      wantedBy = mkIf cfg.autostart ["multi-user.target"];
      after = ["network.target" "elodin-db.service"];
      wants = ["elodin-db.service"];

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/msp-osd --config ${configPath} --mode ${cfg.mode} ${concatStringsSep " " cfg.extraArgs}";
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
