{lib, ...}: {
  networking.hostName = lib.mkForce "";
  systemd.services.set-hostname = {
    description = "Set hostname from /proc/device-tree/serial-number";
    before = ["systemd-hostnamed.service"];
    serviceConfig = {
      User = "root";
      Group = "root";
      Type = "oneshot";
      RemainAfterExit = true;
    };
    script = ''
      SERIAL_SUFFIX=$(cat /proc/device-tree/serial-number | tr -d '\0' | xargs printf '%x' | tail -c 4)
      echo "aleph-$SERIAL_SUFFIX" > /etc/hostname
      echo "Hostname set to aleph-$SERIAL_SUFFIX"
    '';
    wantedBy = ["sysinit.target"];
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
