{...}: {
  boot.kernelParams = [
    "g_ncm.dev_addr=bf:02:bf:e2:b4:90"
    "g_ncm.host_addr=cf:52:68:62:d9:08"
  ];
  networking.firewall.enable = false;
  networking.interfaces.usb0.useDHCP = false;
  networking.interfaces.usb0.ipv4.addresses = [
    {
      address = "10.224.0.1";
      prefixLength = 24;
    }
  ];
  services.kea.dhcp4 = {
    enable = true;
    settings = {
      interfaces-config.interfaces = ["usb0"];
      lease-database = {
        name = "/var/lib/kea/dhcp4.leases";
        persist = true;
        type = "memfile";
      };
      rebind-timer = 2000;
      renew-timer = 1000;
      subnet4 = [
        {
          pools = [
            {
              pool = "10.224.0.2 - 10.224.0.254";
            }
          ];
          subnet = "10.224.0.0/24";
        }
      ];
      valid-lifetime = 4000;
    };
  };
  services.avahi = {
    enable = true;
    publish = {
      enable = true;
      userServices = true;
      addresses = true;
    };
  };
}
