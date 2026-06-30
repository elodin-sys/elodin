# RAM-resident variant of the USB sd-image installer.
#
# The normal installer mounts its ext4 root from the USB drive, which fails on
# units where USB-mass-storage enumeration is unreliable. This module embeds the
# whole Nix store in the initrd (NixOS netboot / live-media style) so the
# installer runs entirely from RAM: UEFI loads the kernel + (large) initrd from
# the ESP and nothing needs a USB block device. Once booted, run
# `aleph-installer` to write the system to NVMe.
#
# Only imported by the `ram-installer` configuration (see flake.nix).
{
  config,
  lib,
  modulesPath,
  ...
}: {
  imports = [(modulesPath + "/installer/netboot/netboot.nix")];

  # netboot.nix and sd-image.nix (via fs.nix) both define the generic image
  # outputs; we flash the sd-image (USB) artifact, so pin these to the sd-image
  # values to resolve the duplicate definitions.
  image.fileName = lib.mkForce "aleph-ram-installer.img";
  image.extension = lib.mkForce "img";
  image.filePath = lib.mkForce "sd-image/${config.image.fileName}";
  system.build.image = lib.mkForce config.system.build.sdImage;
  sdImage.compressImage = lib.mkForce false;

  # Both hardware.nix and netboot.nix disable grub; resolve the duplicate def.
  boot.loader.grub.enable = lib.mkForce false;

  # The entire store ships inside the initrd, so the ESP/firmware partition must
  # be large enough to hold the netboot ramdisk. Tune this if the image build
  # reports the firmware partition is too small.
  sdImage.firmwareSize = lib.mkForce 4096; # MiB

  # The store lives in the initrd squashfs; root is a tmpfs and the sd-image ext4
  # root partition is never mounted. By default sd-image writes the whole system
  # closure into that (unused) ext4 partition — a second ~GB-scale copy of the
  # closure that roughly doubles build time and image size. Leave it empty.
  sdImage.storePaths = lib.mkForce [];

  # hardware.nix uses `lib.mkForce` on boot.initrd.availableKernelModules, which
  # drops netboot's squashfs/overlay entries. Force-load the live-media modules
  # via kernelModules (which hardware.nix does not override) so they are present
  # in the initrd regardless.
  boot.initrd.kernelModules = ["squashfs" "overlay" "loop"];

  # Boot the store-in-initrd ramdisk from the ESP instead of the bootspec initrd
  # so nothing needs to mount a USB block device.
  aleph.espInitrd = "${config.system.build.netbootRamdisk}/initrd";

  # This is a fully offline installer (the whole closure ships in the initrd).
  # `nixos-install` copies the system into /mnt by substituting from the local
  # store (it passes `--extra-substituters auto?trusted=1`). Drop the network
  # binary caches so that substitution doesn't emit noisy "could not resolve
  # host" retries for cache.nixos.org / the Elodin cache. NOTE: do not disable
  # substitution entirely (`substitute false`) — nixos-install needs the local
  # `auto` substituter to populate /mnt.
  nix.settings.substituters = lib.mkForce [];
  nix.settings.extra-substituters = lib.mkForce [];

  # The squashfs store + nix-path-registration are built from the closure of
  # `netboot.storeContents` (see nixos/lib/make-squashfs.nix). netboot.nix only
  # lists the live system by default, so the system we install to NVMe is not
  # guaranteed to be present/registered in the offline store — making
  # `nixos-install` try (and fail) to fetch it from the network. Add the install
  # target explicitly so the whole thing works offline.
  netboot.storeContents = lib.mkForce [
    config.system.build.toplevel
    config.aleph.installer.system
  ];

  # Both netboot.nix and sd-image.nix (via fs.nix) define a `register-nix-paths`
  # service; merged, sd-image's `ConditionPathExists = /nix-path-registration`
  # wins. That path only exists on the sd-image's ext4 root — in this
  # store-in-initrd image the registration is at /nix/store/nix-path-registration
  # — so the service is SKIPPED, the Nix DB is never populated before nix-daemon
  # starts, and offline `nixos-install` treats the (present) install target as
  # missing and tries the network. Replace it with one clean definition that
  # loads the correct file *before* the daemon (so the daemon sees the paths;
  # a late `nix-store --load-db` doesn't help because the daemon caches validity).
  systemd.services.register-nix-paths = lib.mkForce {
    description = "Register Nix Store Paths";
    unitConfig.DefaultDependencies = false;
    wantedBy = ["sysinit.target"];
    before = ["sysinit.target" "shutdown.target" "nix-daemon.socket" "nix-daemon.service"];
    after = ["local-fs.target"];
    conflicts = ["shutdown.target"];
    restartIfChanged = false;
    serviceConfig = {
      Type = "oneshot";
      RemainAfterExit = true;
    };
    script = ''
      ${lib.getExe' config.nix.package "nix-store"} --load-db < /nix/store/nix-path-registration
      touch /etc/NIXOS
      ${lib.getExe' config.nix.package "nix-env"} -p /nix/var/nix/profiles/system --set /run/current-system
    '';
  };
}
