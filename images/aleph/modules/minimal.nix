{lib, ...}: {
  #nix.enable = false;
  # nixpkgs.overlays = [
  #   (
  #     self: super: {
  #       dbus = super.dbus.override {
  #         systemdMinimal = self.systemd;
  #       };
  #     }
  #   )
  #   (
  #     self: super: {
  #       fuse3 = (self.lib.dontRecurseIntoAttrs (self.callPackage (nixpkgs.outPath + "/pkgs/os-specific/linux/fuse") {})).fuse_3;
  #     }
  #   )
  # ];
  #services.udev.enable = false;
  #services.lvm.enable = false;
  system.switch.enable = false;
  system.switch.enableNg = true;

  # changes sourced from: https://github.com/NixOS/nixpkgs/blob/master/nixos/modules/profiles/perlless.nix
  boot.initrd.systemd.enable = true;
  boot.initrd.systemd.enableTpm2 = false;
  # system.etc.overlay.enable = lib.mkDefault true; # renable once we are on upstream kernel
  # services.userborn.enable = lib.mkDefault true;

  # Random perl remnants
  system.disableInstallerTools = lib.mkDefault false;
  programs.less.lessopen = lib.mkDefault null;
  programs.command-not-found.enable = lib.mkDefault false;
  boot.enableContainers = lib.mkDefault false;
  documentation.info.enable = lib.mkDefault false;
}
