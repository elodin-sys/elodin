{
  pkgs,
  lib,
  config,
  ...
}: let
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
  };

  config = lib.mkIf cfg.enable {
    systemd.services.elodin-db = with pkgs; {
      wantedBy = ["multi-user.target"];
      after = ["network.target"];
      description = "start elodin-db";
      serviceConfig = {
        Type = "exec";
        User = "root";
        ExecStart = "${elodin-db}/bin/elodin-db run [::]:2240 --http-addr [::]:2248 /db";
        KillSignal = "SIGINT";
        Environment = "RUST_LOG=info";
      };
    };
    environment.systemPackages = [pkgs.elodin-db];
    networking.firewall.allowedTCPPorts = lib.optionals cfg.openFirewall [2240 2248];
  };
}
