{
  pkgs,
  lib,
  config,
  ...
}: let
  elodin-db = pkgs.elodin.elodin-db.bin;
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
    systemd.services."elodin-db@" = with pkgs; {
      after = ["network.target"];
      description = "start elodin-db";
      serviceConfig = {
        Type = "exec";
        User = "root";
        ExecStart = "${elodin-db}/bin/elodin-db run [::]:2240 --http-addr [::]:2248 ${cfg.dbFolderName}/%i";
        KillSignal = "SIGINT";
        Environment = "RUST_LOG=info";
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
