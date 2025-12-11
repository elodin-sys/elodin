{
  lib,
  pkgs,
  config,
  ...
}: let
  overlay = final: prev: let
    kernel = final.callPackage ../kernel-r36/default.nix {
      structuredExtraConfig = with lib.kernel; {
        USB_GADGET = lib.mkForce yes;
        USB_G_NCM = lib.mkForce yes;
        INET = yes;
        DUMMY = yes;
        IPVLAN = module;

        R8169 = module;

        # NVMe support for root drive
        BLK_DEV_NVME = module;

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

        # More USB
      };
      l4t-xusb-firmware = final.nvidia-jetpack.l4t-xusb-firmware;
      kernelPatches = [];
      inherit (final.nvidia-jetpack) gitRepos;
      inherit (final.nvidia-jetpack) devicetree;
    };
  in {
    aleph.kernelPackages = (prev.linuxPackagesFor kernel).extend final.nvidia-jetpack.kernelPackagesOverlay;
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
  #sdImage.compressImage = true;
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
  #boot.extraModulePackages = lib.mkForce [];


  # Override kernel modules to exclude modules that don't exist in this kernel version
  boot.initrd.availableKernelModules = lib.mkForce [
    "ahci"
    "xhci_pci"
    "sd_mod"
    "xhci_hcd"
    # USB modules that are available
    "xhci-tegra"
    "ucsi_ccg"
    "typec_ucsi"
    "typec"
    # Storage modules that are available
    "sd_mod"
    "usb_storage"
    # NVMe and PCIe modules for NVMe drive
    "nvme"
    "phy_tegra194_p2u"
    "pcie_tegra194"
    "tps6598x"
    # Filesystem modules
    "ext4"
  ];

  # Avoids a bunch ofe extra modules we don't have in the tegra_defconfig, like "ata_piix",
  disabledModules = ["profiles/all-hardware.nix"];

  hardware.nvidia-jetpack = {
    enable = true;
    majorVersion = "6";
    som = "orin-nx";
    carrierBoard = "devkit";
    #configureCuda = false;
    #kernel.realtime = true;
  };

  #hardware.deviceTree.name = "tegra234-p3767-0000-aleph.dtb";
  # Configure device tree for Orin NX devkit (p3767-0000)
  #hardware.deviceTree.name = "tegra234-p3768-0000+p3767-0000.dtb";
  hardware.deviceTree.name = "tegra234-p3767-0000-aleph.dtb";
  # hardware.deviceTree.overlays = [
  # {
  #   name = "aleph-b2b";
  #   dtboFile = "${config.boot.kernelPackages.devicetree}/tegra234-aleph-b2b.dtbo";
  # }
  # ];

  hardware.firmware = lib.mkAfter [pkgs.linux-firmware];

  # TODO: REVISIT CONSEQUENCE OF DISABLING THIS, MIGHT BE HANDLED UPSTREAM BY DYNAMIC LOGIC
  services.nvpmodel.enable = false;
}
