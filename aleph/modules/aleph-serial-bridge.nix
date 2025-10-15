{pkgs, ...}: {
  systemd.services.serial-bridge = with pkgs; {
    wantedBy = ["multi-user.target"];
    after = ["network.target"];
    description = "start aleph-serial-bridge";
    serviceConfig = {
      Type = "exec";
      User = "root";
      ExecStart = "${serial-bridge}/bin/aleph-serial-bridge";
      KillSignal = "SIGINT";
      Environment = "RUST_LOG=debug";
    };
  };
}
