{pkgs, ...}: {
  systemd.services.mekf = with pkgs; {
    wantedBy = ["multi-user.target"];
    after = ["elodin-db.service"];
    description = "start mekf";
    serviceConfig = {
      Type = "exec";
      User = "root";
      ExecStart = "${mekf}/bin/mekf";
      KillSignal = "SIGINT";
      Environment = "RUST_LOG=info";
    };
  };
}
