{
  config,
  lib,
  pkgs,
  ...
}:
with lib; let
  cfg = config.services.udp-component-receive;
in {
  options.services.udp-component-receive = {
    enable = mkEnableOption "UDP Component Receive service for Elodin-DB";

    autostart = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Whether to auto-start the UDP Component Receive service.
        When false, the service is configured but not started automatically.
        Use `systemctl start udp-component-receive` to start manually.
      '';
    };

    package = mkOption {
      type = types.package;
      default = pkgs.udp-component-broadcast;
      description = "UDP Component Broadcast package to use (contains both broadcaster and receiver)";
    };

    dbAddr = mkOption {
      type = types.str;
      default = "127.0.0.1:2240";
      description = "Elodin-DB address to write received components to";
    };

    listenPort = mkOption {
      type = types.port;
      default = 41235;
      description = "UDP port to listen for broadcasts on";
    };

    filter = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Only accept specific component names (empty list accepts all)";
      example = ["target.world_pos" "target.world_vel"];
    };

    verbose = mkOption {
      type = types.bool;
      default = false;
      description = "Enable verbose logging";
    };

    timestampMode = mkOption {
      type = types.enum ["sender" "local" "monotonic"];
      default = "sender";
      description = ''
        Timestamp mode for received components:
        - sender: Use timestamp from broadcaster (default)
        - local: Use local wall-clock time
        - monotonic: Use Linux monotonic clock
      '';
    };

    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Extra command-line arguments to pass to the receiver";
    };
  };

  config = mkIf cfg.enable {
    # Add package to system packages for manual debugging
    environment.systemPackages = [cfg.package];

    # Open firewall for UDP broadcast reception
    networking.firewall.allowedUDPPorts = [cfg.listenPort];

    systemd.services.udp-component-receive = {
      description = "UDP Component Receive Service for Elodin-DB";
      wantedBy = mkIf cfg.autostart ["multi-user.target"];
      after = ["network.target" "elodin-db.service"];
      wants = ["elodin-db.service"];

      serviceConfig = {
        Type = "simple";
        ExecStart = let
          filterArgs = concatMapStringsSep " " (f: "--filter ${f}") cfg.filter;
          verboseArg = optionalString cfg.verbose "-v";
          timestampArg = "--timestamp-mode ${cfg.timestampMode}";
        in "${cfg.package}/bin/udp-receive --db-addr ${cfg.dbAddr} --listen-port ${toString cfg.listenPort} ${timestampArg} ${filterArgs} ${verboseArg} ${concatStringsSep " " cfg.extraArgs}";
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
