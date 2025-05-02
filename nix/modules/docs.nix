{
  pkgs,
  lib,
  config,
  ...
}: let
  content = pkgs.stdenv.mkDerivation {
    name = "docs-content";
    src = ../../docs/public;

    buildInputs = [pkgs.zola];
    buildPhase = "zola build";

    installPhase = ''
      mkdir -p $out
      cp -r ./public/* $out/
    '';
  };
  cfg = config.services.elodin-docs;
in {
  options.services.elodin-docs = {
    enable = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to enable the elodin-docs service.
      '';
    };
    port = lib.mkOption {
      type = lib.types.port;
      default = 80;
      description = ''
        Specifies on which port the elodin-docs service listens.
      '';
    };
    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to automatically open the specified ports in the firewall.
      '';
    };
  };
  config = lib.mkIf cfg.enable {
    # Create a dedicated user and group for the service
    users.users.elodin-docs = {
      isSystemUser = true;
      group = "elodin-docs";
      description = "Elodin Documentation Server user";
      home = "/var/empty";
    };
    users.groups.elodin-docs = {};

    systemd.services.elodin-docs = {
      description = "Elodin Documentation Server";
      wantedBy = ["multi-user.target"];
      after = ["network.target"];
      serviceConfig = {
        # Basic service settings
        User = "elodin-docs";
        Group = "elodin-docs";
        ExecStart = "${pkgs.memserve}/bin/memserve --bind-address [::]:${toString cfg.port} --log-level debug";
        WorkingDirectory = "${content}";
        Restart = "on-failure";
        RestartSec = "5s";

        # Resource limits
        CPUQuota = "100%"; # Allow up to 1 full core
        MemoryMax = "2G"; # Allow up to 2GB of memory
        TasksMax = "1"; # Restrict to exactly 1 process (no threads)
        LimitNPROC = 1; # Prevent any child process spawning

        # High-value security hardening (removing defaults and low-value options)
        CapabilityBoundingSet = "CAP_NET_BIND_SERVICE";
        NoNewPrivileges = true;
        PrivateDevices = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadOnlyPaths = "${content}";
        RestrictAddressFamilies = "AF_INET AF_INET6";
        SystemCallFilter = "@system-service";
        # Always add the capability for binding to privileged ports
        AmbientCapabilities = "CAP_NET_BIND_SERVICE";
      };
    };

    environment.systemPackages = [
      pkgs.memserve
    ];
    networking.firewall.allowedTCPPorts = lib.optionals cfg.openFirewall [cfg.port];
  };
}
