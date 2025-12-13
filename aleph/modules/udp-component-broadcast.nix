{
  config,
  lib,
  pkgs,
  ...
}:
with lib; let
  cfg = config.services.udp-component-broadcast;
in {
  options.services.udp-component-broadcast = {
    enable = mkEnableOption "UDP Component Broadcast service for Elodin-DB";

    autostart = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Whether to auto-start the UDP Component Broadcast service.
        When false, the service is configured but not started automatically.
        Use `systemctl start udp-component-broadcast` to start manually.
      '';
    };

    package = mkOption {
      type = types.package;
      default = pkgs.udp-component-broadcast;
      description = "UDP Component Broadcast package to use";
    };

    component = mkOption {
      type = types.str;
      description = "Component name to subscribe to and broadcast (e.g., bdx.world_pos)";
      example = "bdx.world_pos";
    };

    rename = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Rename component for broadcast (e.g., target.world_pos). If null, uses the original component name.";
      example = "target.world_pos";
    };

    sourceId = mkOption {
      type = types.str;
      default = "source";
      description = "Source identifier for this broadcaster";
    };

    dbAddr = mkOption {
      type = types.str;
      default = "127.0.0.1:2240";
      description = "Elodin-DB address to subscribe from";
    };

    broadcastRate = mkOption {
      type = types.float;
      default = 10.0;
      description = "Broadcast rate in Hz";
    };

    broadcastPort = mkOption {
      type = types.port;
      default = 41235;
      description = "UDP broadcast port";
    };

    verbose = mkOption {
      type = types.bool;
      default = false;
      description = "Enable verbose logging";
    };

    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Extra command-line arguments to pass to the broadcaster";
    };
  };

  config = mkIf cfg.enable {
    # Add package to system packages for manual debugging
    environment.systemPackages = [cfg.package];

    systemd.services.udp-component-broadcast = {
      description = "UDP Component Broadcast Service for Elodin-DB";
      wantedBy = mkIf cfg.autostart ["multi-user.target"];
      after = ["network.target" "elodin-db.service"];
      wants = ["elodin-db.service"];

      serviceConfig = {
        Type = "simple";
        ExecStart = let
          renameArg = optionalString (cfg.rename != null) "--rename ${cfg.rename}";
          verboseArg = optionalString cfg.verbose "-v";
        in "${cfg.package}/bin/udp-broadcast --component ${cfg.component} ${renameArg} --db-addr ${cfg.dbAddr} --broadcast-rate ${toString cfg.broadcastRate} --broadcast-port ${toString cfg.broadcastPort} --source-id ${cfg.sourceId} ${verboseArg} ${concatStringsSep " " cfg.extraArgs}";
        Restart = "always";
        RestartSec = 5;
        StandardOutput = "journal";
        StandardError = "journal";

        # Run as root (can be changed if needed)
        User = "root";

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
  };
}
