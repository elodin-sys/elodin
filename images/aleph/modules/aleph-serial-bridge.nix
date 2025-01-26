{pkgs, ...}: {
  systemd.services.file-bridge = with pkgs; {
    wantedBy = ["multi-user.target"];
    after = ["network.target"];
    description = "start aleph-file-bridge";
    serviceConfig = {
      Type = "exec";
      User = "root";
      ExecStart = "${serial-bridge}";
      KillSignal = "SIGINT";
      Environment = "RUST_LOG=debug";
    };
  };
}
