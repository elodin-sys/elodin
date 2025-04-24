{lib, ...}: {
  networking.hostName = lib.mkForce "";
  systemd.services.set-hostname = {
    description = "Set hostname from /etc/machine-id";
    after = ["network.target"];
    before = ["systemd-hostnamed.service"];
    serviceConfig = {
      User = "root";
      Group = "root";
      Type = "oneshot";
      RemainAfterExit = true;
    };
    script = ''
      ID_PREFIX=$(head -c 4 /etc/machine-id)
      echo "aleph-$ID_PREFIX" > /etc/hostname
      echo "Hostname set to aleph-$ID_PREFIX"
    '';
    wantedBy = ["multi-user.target"];
  };
  networking.dhcpcd.enable = true;
  networking.wireless.iwd = {
    enable = true;
    settings = {
      IPv6 = {
        Enabled = true;
      };
      Settings = {
        AutoConnect = true;
      };
    };
  };
  boot.extraModprobeConfig = ''
    options iwlwifi power_save=0 uapsd_disable=1 d0i3_disable=1
    options iwlmvm power_scheme=1
  '';
}
