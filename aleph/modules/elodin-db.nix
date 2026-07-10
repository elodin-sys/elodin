{
  pkgs,
  lib,
  config,
  ...
}: let
  elodin-db = pkgs.elodin-db;
  cfg = config.services.elodin-db;
  assetsFlag = lib.optionalString (cfg.assetsDir != null) " --assets ${cfg.assetsDir}";

  # The shared asset root is normally created (tmpfiles) and populated
  # (elodin-assets-seed.service) by the `elodin` module. Those only exist when
  # that module is enabled, so elodin-db must not silently rely on them: when it
  # runs standalone (e.g. services.elodin.enable = false) it has to create its
  # own ingest source, and it must only order after a seed unit that exists.
  # `or false` keeps this valid even if the `elodin` module is not imported.
  elodinEnabled = config.services.elodin.enable or false;
  elodinSeeds = elodinEnabled && (config.services.elodin.examples or false);
  # The `elodin` module hardcodes this path; when our assetsDir matches it and
  # that module is enabled, defer to it for directory creation to avoid a
  # duplicate tmpfiles entry. Otherwise we own the directory.
  elodinOwnsAssetsDir = elodinEnabled && cfg.assetsDir == "/var/lib/elodin/assets";
  assetsAfter = lib.optional elodinSeeds "elodin-assets-seed.service";
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
    assetsDir = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = "/var/lib/elodin/assets";
      description = ''
        Source assets/ tree ingested into each fresh database on creation
        (passed to `elodin-db run --assets`). Defaults to the shared asset
        root seeded by the elodin module, so every recorded database carries
        its schematic assets and is a complete, portable record. Set to null
        to disable ingest.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services."elodin-db@" = {
      # Order after the asset seed (from the elodin module, when present) so a
      # fresh boot ingests a fully populated tree, not a partial one.
      after = ["network.target"] ++ assetsAfter;
      stopIfChanged = false;
      restartIfChanged = false;
      description = "Start elodin-db under the folder '%i'";
      serviceConfig = {
        Type = "exec";
        User = "root";
        ExecStart = "${elodin-db}/bin/elodin-db run [::]:2240 --http-addr [::]:2248${assetsFlag} ${cfg.dbFolderName}/%i";
        KillSignal = "SIGINT";
        Environment = "RUST_LOG=info";
      };
    };

    systemd.services.elodin-db = lib.mkIf (cfg.autostart && !cfg.dbUniqueOnBoot) {
      after = ["network.target"] ++ assetsAfter;
      wantedBy = ["multi-user.target"];
      description = "Elodin-DB telemetry database";
      serviceConfig = {
        Type = "exec";
        User = "root";
        ExecStart = "${elodin-db}/bin/elodin-db run [::]:2240 --http-addr [::]:2248${assetsFlag} ${cfg.dbFolderName}/default";
        KillSignal = "SIGINT";
        Restart = "on-failure";
        RestartSec = "5s";
        Environment = "RUST_LOG=info";
      };
    };

    systemd.services."elodin-db-default" = lib.mkIf (cfg.autostart && cfg.dbUniqueOnBoot) {
      after = ["network.target"] ++ assetsAfter;
      wantedBy = ["multi-user.target"];
      stopIfChanged = false;
      restartIfChanged = false;
      description = "Start a unique elodin-db instance for this boot";
      serviceConfig = let
        stopScript = pkgs.writeShellScript "elodin-db-stop-old" ''
          export PATH="${lib.makeBinPath [pkgs.coreutils pkgs.gawk pkgs.iproute2 pkgs.systemd]}:$PATH"
          for unit in $(systemctl list-units --type=service --state=active --plain --no-legend 'elodin-db@*' | awk '{print $1}'); do
            echo "Stopping $unit"
            systemctl stop "$unit" || true
          done
          # Wait for port 2240 to be free (up to 10 seconds)
          for i in $(seq 1 20); do
            if ! ss -tlnp | grep -q ':2240 '; then
              break
            fi
            sleep 0.5
          done
        '';
      in {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStartPre = stopScript;
        ExecStop = stopScript;
        ExecStart = pkgs.writeShellScript "elodin-db-default" ''
          TIMESTAMP=$(date +%Y%m%d-%H%M%S)
          systemctl start "elodin-db@default-$TIMESTAMP.service"
        '';
      };
    };

    system.activationScripts.restartElodinDb = lib.mkIf (cfg.autostart && cfg.dbUniqueOnBoot) {
      text = ''
        if [ -d /run/systemd/system ] && \
           /run/current-system/sw/bin/systemctl is-active --quiet elodin-db-default.service 2>/dev/null; then
          echo "restarting elodin-db-default for fresh database..."
          /run/current-system/sw/bin/systemctl restart elodin-db-default.service || true
        fi
      '';
    };

    # Ensure the ingest source exists so `--assets` never points at a missing
    # directory (ingest would otherwise warn and the DB would start without its
    # asset tree). When the `elodin` module owns this same path we defer to it.
    # A group-writable setgid dir mirrors the shared ELODIN_ASSETS convention so
    # wheel users can drop in their own assets.
    systemd.tmpfiles.rules = lib.optionals (cfg.assetsDir != null && !elodinOwnsAssetsDir) [
      "d ${cfg.assetsDir} 2775 root wheel - -"
    ];

    environment.systemPackages = [elodin-db];
    networking.firewall.allowedTCPPortRanges = lib.optionals cfg.openFirewall [
      {
        from = 1;
        to = 65535;
      }
    ];
    networking.firewall.allowedUDPPortRanges = lib.optionals cfg.openFirewall [
      {
        from = 1;
        to = 65535;
      }
    ];
  };
}
