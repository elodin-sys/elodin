{
  pkgs,
  lib,
  config,
  ...
}: let
  elodin-db = pkgs.elodin-db;
  cfg = config.services.elodin-db;
  elodin-db-wrapper = pkgs.writeShellScriptBin "elodin-db-wrapper" ''
    DB_NAME="$1"
    if [ "$DB_UNIQUE_ON_BOOT" = "1" ]; then
      TIMESTAMP=$(date +%Y%m%d-%H%M%S)
      DB_PATH="${cfg.dbFolderName}/$DB_NAME-$TIMESTAMP"
    else
      DB_PATH="${cfg.dbFolderName}/$DB_NAME"
    fi
    exec ${elodin-db}/bin/elodin-db run [::]:2240 --http-addr [::]:2248 "$DB_PATH"
  '';
in {
  options.services.elodin-db = {
    enable = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to enable the elodin-db service.
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
    dbUniqueOnBoot = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Whether to automatically create a unique db on boot. This is useful if you are using a different time source (such as CLOCK_MONOTONIC).
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services."elodin-db@" = with pkgs; {
      after = ["network.target"];
      description = "start elodin-db";
      serviceConfig = {
        Type = "exec";
        User = "root";
        # Our desired run command is `elodin-db run [::]:2240 --http-addr [::]:2248 /db/default"`
        # but we wrap it in a shell script that allows a timestamp to be appended to the path.
        # For design motivation, see the description of: services.elodin-db.dbUniqueOnBoot
        ExecStart = "${elodin-db-wrapper}/bin/elodin-db-wrapper %i";
        KillSignal = "SIGINT";
        Environment = [
          "RUST_LOG=info"
          "DB_UNIQUE_ON_BOOT=${
            if cfg.dbUniqueOnBoot
            then "1"
            else "0"
          }"
        ];
      };
    };

    systemd.packages = [
      (pkgs.runCommandNoCC "elodin-db-default-service" {
          preferLocalBuild = true;
          allowSubstitutes = false;
        } ''
          mkdir -p $out/etc/systemd/system/
          ln -s /etc/systemd/system/elodin-db@.service $out/etc/systemd/system/elodin-db@default.service
        '')
    ];

    # see: https://github.com/NixOS/nixpkgs/issues/80933
    systemd.services."elodin-db@default" = {
      wantedBy = ["multi-user.target"];
      overrideStrategy = "asDropin";
    };

    environment.systemPackages = [elodin-db];
    networking.firewall.allowedTCPPorts = lib.optionals cfg.openFirewall [2240 2248];
  };
}
