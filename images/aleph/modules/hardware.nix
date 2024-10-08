{
  lib,
  pkgs,
  ...
}: let
  overlay = final: prev: let
    kernel = prev.callPackage ../kernel/default.nix {
      structuredExtraConfig = with lib.kernel; {
        USB_GADGET = lib.mkForce yes;
        USB_G_NCM = lib.mkForce yes;
        INET = yes;
      };
      l4t-xusb-firmware = prev.nvidia-jetpack.l4t-xusb-firmware;
      kernelPatches = [];
    };
  in {
    aleph.kernelPackages = prev.linuxPackagesFor kernel;
  };
in {
  nixpkgs.overlays = [overlay];
  boot.loader.grub.enable = false;
  boot.loader.generic-extlinux-compatible.enable = true;
  boot.loader.generic-extlinux-compatible.useGenerationDeviceTree = true;
  boot.kernelPackages = lib.mkForce pkgs.aleph.kernelPackages;
  boot.kernelParams = [
    "console=tty0"
    "fbcon=map:0"
    "video=efifb:off"
    "console=ttyTCU0,115200"
    "nohibernate"
    "loglevel=4"
  ];
  boot.extraModulePackages = lib.mkForce [];

  # Avoids a bunch ofeextra modules we don't have in the tegra_defconfig, like "ata_piix",
  disabledModules = ["profiles/all-hardware.nix"];
  #hardware.deviceTree.name = "tegra234-p3767-0003-p3509-a02.dtb";
  hardware.deviceTree.name = "tegra234-p3767-0004-antmicro-job.dtb";
  hardware.nvidia-jetpack = {
    enable = true;
    som = "orin-nx";
    sku = "0001";
    carrierBoard = "devkit";
    #kernel.realtime = true;
  };
  hardware.firmware = [pkgs.linux-firmware];
  system.activationScripts.extlinux-fixed-path.text = ''
    ${pkgs.gnused}/bin/sed -i 's/\.\.\/nixos/\/boot\/nixos/g' /boot/extlinux/extlinux.conf # Jetson doesn't like relative paths
  '';
}
