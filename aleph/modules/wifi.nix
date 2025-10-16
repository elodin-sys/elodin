{
  lib,
  pkgs,
  ...
}: let
  setHostnameScript = pkgs.writeShellScriptBin "set-hostname" ''
    #!/bin/sh -e
    SERIAL_SUFFIX=$(cat /proc/device-tree/serial-number | tr -d '\0' | xargs printf '%x' | tail -c 4)
    echo "aleph-$SERIAL_SUFFIX" > /etc/hostname
    ${pkgs.nettools}/bin/hostname aleph-$SERIAL_SUFFIX
    echo "Hostname set to aleph-$SERIAL_SUFFIX"
  '';
in {
  networking.hostName = lib.mkForce "";
  environment.systemPackages = [
    setHostnameScript
  ];
  systemd.services.set-hostname = {
    description = "Set hostname from /proc/device-tree/serial-number";
    serviceConfig = {
      User = "root";
      Group = "root";
      Type = "oneshot";
      RemainAfterExit = true;
      ExecStart = "${setHostnameScript}/bin/set-hostname";

      ExecCondition = "${pkgs.coreutils}/bin/test -e /proc/device-tree/serial-number";
      RestartSec = "100ms";
      Restart = "on-failure";
    };
    before = ["network.target" "avahi-daemon.service"];
    wantedBy = ["network-pre.target"];
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
  # Disable all the power saving features. They all negatively impact reliability.
  boot.extraModprobeConfig = ''
    options iwlwifi power_save=0 uapsd_disable=1 d0i3_disable=1
    options iwlmvm power_scheme=1
  '';
}
