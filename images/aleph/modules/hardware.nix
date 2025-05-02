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

        # Add these crypto modules needed by iwd
        CRYPTO = yes;
        CRYPTO_USER = yes;
        CRYPTO_USER_API = yes;
        CRYPTO_USER_API_HASH = yes;
        CRYPTO_USER_API_SKCIPHER = yes;
        CRYPTO_MD5 = yes;
        CRYPTO_SHA1 = yes;
        CRYPTO_SHA256 = yes;
        CRYPTO_SHA512 = yes;
        CRYPTO_AES = yes;
        CRYPTO_CMAC = yes;
        CRYPTO_HMAC = yes;

        # ARM-specific optimized implementations for Orin
        CRYPTO_SHA1_ARM64_CE = yes;
        CRYPTO_SHA2_ARM64_CE = yes;
        CRYPTO_SHA512_ARM64_CE = yes;
        CRYPTO_AES_ARM64_CE = yes;
        CRYPTO_GHASH_ARM64_CE = yes;
      };
      l4t-xusb-firmware = prev.nvidia-jetpack.l4t-xusb-firmware;
      kernelPatches = [];
    };
  in {
    aleph.kernelPackages = prev.linuxPackagesFor kernel;
  };
in {
  nixpkgs.overlays = [
    overlay
    (final: prev: {
      systemd = prev.systemd.overrideAttrs (prevAttrs: {
        patches =
          prevAttrs.patches
          ++ [
            ./systemd-boot-double-dtb-buffer-size.patch
          ];
      });
      systemd-minimal = prev.systemd-minimal.overrideAttrs (prevAttrs: {
        patches =
          prevAttrs.patches
          ++ [
            ./systemd-boot-double-dtb-buffer-size.patch
          ];
      });
    })
  ];
  imports = [
    ./systemd-boot-dtb.nix
  ];
  sdImage.compressImage = true;
  boot.loader.systemd-boot.enable = true;
  boot.loader.systemd-boot.installDeviceTree = true;
  boot.loader.systemd-boot-dtb.enable = true;
  boot.loader.efi.canTouchEfiVariables = false;
  boot.loader.grub.enable = false;
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
  hardware.deviceTree.name = "tegra234-p3767-0000-aleph.dtb";
  hardware.nvidia-jetpack = {
    enable = true;
    som = "orin-nx";
    carrierBoard = "devkit";
    #kernel.realtime = true;
  };
  hardware.firmware = [pkgs.linux-firmware];
}
