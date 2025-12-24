{lib, ...}: {
  # Add double-dtb-buffer-size patch to systemd-boot to provide
  # space for the addition of larger device tree blobs during boot
  # as of 2026-01 this is required for booting over usb
  # ref: https://github.com/anduril/jetpack-nixos/issues/111#issuecomment-2054146506
  nixpkgs.overlays = [
    (final: prev: let
      patch = ./systemd-boot-double-dtb-buffer-size.patch;
      addPatch = pkg:
        pkg.overrideAttrs (old: {
          patches = (old.patches or []) ++ [patch];
        });
    in {
      systemd = addPatch prev.systemd;
      systemd-minimal = addPatch prev.systemd-minimal;
    })
  ];
  imports = [
    ./systemd-boot-dtb.nix
  ];
  boot.loader.systemd-boot-dtb.enable = true;
  # End systemd-boot-dtb patch

  boot.loader.systemd-boot.enable = true;
  boot.loader.systemd-boot.installDeviceTree = true;
  boot.loader.efi.canTouchEfiVariables = false;
  boot.loader.grub.enable = false;

  boot.kernelParams = [
    "console=tty0"
    "fbcon=map:0"
    "video=efifb:off"
    "console=ttyTCU0,115200"
    "nohibernate"
    "loglevel=4"
  ];

  # Kernel modules to include for the initramfs
  boot.initrd.availableKernelModules = lib.mkForce [
    # Tegra-specific modules for NVMe boot
    "phy_tegra194_p2u"
    "pcie_tegra194"
    # Tegra-specific modules required for usb/sdimage boot
    "xhci-tegra"
  ];

  # Avoids a bunch of extra modules we don't have in the tegra_defconfig, like "ata_piix",
  disabledModules = ["profiles/all-hardware.nix"];

  hardware.nvidia-jetpack = {
    enable = true;
    majorVersion = "6";
    som = "orin-nx";
    carrierBoard = "devkit";
    kernel.realtime = true;
  };

  # Configure device tree for Orin NX devkit (p3767-0000)
  hardware.deviceTree.name = "tegra234-p3767-0000-aleph.dtb";
}
