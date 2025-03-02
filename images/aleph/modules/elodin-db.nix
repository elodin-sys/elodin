{pkgs, ...}: {
  systemd.services.elodin-db = with pkgs; {
    wantedBy = ["multi-user.target"];
    after = ["network.target"];
    description = "start elodin-db";
    serviceConfig = {
      Type = "exec";
      User = "root";
      ExecStart = "${elodin-db}/bin/elodin-db run [::]:2240 /db";
      KillSignal = "SIGINT";
      Environment = "RUST_LOG=info";
    };
  };
}
