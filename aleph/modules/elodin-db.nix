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

    systemd.services.elodin-db = lib.mkIf (cfg.autostart && !cfg.dbUniqueOnBoot) {
      after = ["network.target"];
      wantedBy = ["multi-user.target"];
      description = "Elodin-DB telemetry database";
      serviceConfig = {
        Type = "exec";
        User = "root";
        ExecStart = "${elodin-db}/bin/elodin-db run [::]:2240 --http-addr [::]:2248 ${cfg.dbFolderName}/default";
        KillSignal = "SIGINT";
        Restart = "on-failure";
        RestartSec = "5s";
        Environment = "RUST_LOG=info";
      };
    };

    systemd.services."elodin-db-default" = lib.mkIf (cfg.autostart && cfg.dbUniqueOnBoot) {
      after = ["network.target"];
      wantedBy = ["multi-user.target"];
      restartIfChanged = true;
      description = "Start a unique elodin-db instance for this boot";
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStartPre = pkgs.writeShellScript "elodin-db-stop-old" ''
          export PATH="${lib.makeBinPath [pkgs.coreutils pkgs.gawk pkgs.systemd]}:$PATH"
          for unit in $(systemctl list-units --type=service --plain --no-legend 'elodin-db@*' | awk '{print $1}'); do
            systemctl stop "$unit" 2>/dev/null || true
          done
          sleep 1
        '';
        ExecStart = pkgs.writeShellScript "elodin-db-default" ''
          TIMESTAMP=$(date +%Y%m%d-%H%M%S)
          systemctl start "elodin-db@default-$TIMESTAMP.service"
        '';
      };
    };

    environment.systemPackages = [elodin-db];
    networking.firewall.allowedTCPPorts = lib.optionals cfg.openFirewall [2240 2248];
  };
}
