{
  pkgs,
  lib,
  config,
  ...
}: let
  elodin-db = pkgs.elodin-db;
  cfg = config.services.elodin-db;
in {
  options.services.elodin-db = {
    enable = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to enable the elodin-db service.
      '';
    };
    autostart = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to auto-start the elodin-db service.
      '';
    };
    dbUniqueOnBoot = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to automatically create a unique db on boot. This is useful if you are using a different time source (such as CLOCK_MONOTONIC).
      '';
    };
    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to automatically open the specified ports in the firewall.
      '';
    };
    dbFolderName = lib.mkOption {
      type = lib.types.str;
      default = "/db";
      description = ''
        The parent path for the elodin-db output directory.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    # Create template elodin-db service
    systemd.services."elodin-db@" = {
      after = ["network.target"];
      description = "Start elodin-db under the folder '%i'";
      serviceConfig = {
        Type = "exec";
        User = "root";
        ExecStart = "${elodin-db}/bin/elodin-db run [::]:2240 --http-addr [::]:2248 ${cfg.dbFolderName}/%i";
        KillSignal = "SIGINT";
        Environment = "RUST_LOG=info";
      };
    };

    systemd.services."elodin-db-default" = lib.mkIf cfg.autostart {
      after = ["network.target"];
      wantedBy = ["multi-user.target"];
      description = "Start the default elodin-db instance";
      serviceConfig = {
        Type = "oneshot";
        ExecStart = pkgs.writeShellScript "elodin-db-default" (
          if cfg.dbUniqueOnBoot
          then ''
            TIMESTAMP=$(date +%Y%m%d-%H%M%S)
            systemctl start "elodin-db@default-$TIMESTAMP.service"
          ''
          else ''
            systemctl start "elodin-db@default.service"
          ''
        );
      };
    };

    environment.systemPackages = [elodin-db];
    networking.firewall.allowedTCPPorts = lib.optionals cfg.openFirewall [2240 2248];
  };
}
