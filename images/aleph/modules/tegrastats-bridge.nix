{pkgs, ...}: {
  systemd.services.tegrastats-bridge = with pkgs; {
    wantedBy = ["multi-user.target"];
    after = ["network.target" "elodin-db.target"];
    description = "start tegrastats-bridge";
    serviceConfig = {
      Type = "exec";
      User = "root";
      ExecStart = "${tegrastats-bridge}/bin/tegrastats-bridge";
      KillSignal = "SIGINT";
      Environment = "RUST_LOG=debug";
    };
  };
}
