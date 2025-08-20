{...}: {
  boot.kernelParams = [
    "g_ncm.dev_addr=ee:a1:ef:ee:a1:ef"
    "g_ncm.host_addr=ee:a1:ef:ee:a1:ef"
  ];
  networking.interfaces.usb0.useDHCP = false;
  networking.interfaces.dummy0.useDHCP = false;
  networking.interfaces.usb0.ipv4.addresses = [
    {
      address = "10.224.0.1";
      prefixLength = 24;
    }
  ];
  networking.interfaces.usb0.ipv6.addresses = [
    {
      address = "fde1:2240:a1ef::1";
      prefixLength = 64;
    }
  ];
  services.radvd = {
    enable = true;
    config = ''
      interface usb0 {
        AdvSendAdvert on;
        MaxRtrAdvInterval 20;
        MinRtrAdvInterval 10;
        prefix fde1:2240:a1ef::/64 {
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
    nssmdns4 = true;
    denyInterfaces = ["dummy0" "lo" "usb0"];
    publish = {
      enable = true;
      userServices = true;
      addresses = true;
      workstation = true;
      hinfo = true;
    };
  };
}
