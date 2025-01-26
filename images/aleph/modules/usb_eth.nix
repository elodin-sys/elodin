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
  networking.interfaces.usb0.ipv6.addresses = [
    {
      address = "fd48:2240:ffff::";
      prefixLength = 64;
    }
  ];
  services.radvd = {
    enable = true;
    config = ''
      interface usb0 {
        AdvSendAdvert on;
        prefix fd48:2240:ffff::/64 {
          AdvAutonomous on;
          AdvRouterAddr off;
          AdvOnLink on;
        };
      };
    '';
  };
  boot.kernel.sysctl = {"net.ipv6.conf.all.forwarding" = true;};
  services.avahi = {
    enable = true;
    publish = {
      enable = true;
      userServices = true;
      addresses = true;
    };
  };
}
