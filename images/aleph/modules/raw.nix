{
  pkgs,
  lib,
  config,
  modulesPath,
  ...
}: {
  fileSystems."/" = {
    device = "/dev/disk/by-label/nixos";
    fsType = "ext4";
    autoResize = true;
  };

  system.build.raw = pkgs.callPackage "${modulesPath}/../lib/make-ext4-fs.nix" {
    storePaths = [config.system.build.toplevel];
    volumeLabel = "nixos";
    populateImageCommands = ''
      mkdir -p ./files/boot
      ${config.boot.loader.generic-extlinux-compatible.populateCmd} -c ${config.system.build.toplevel} -d ./files/boot
      sed -i 's/..\/nixos/\/boot\/nixos/g' ./files/boot/extlinux/extlinux.conf # Jetson doesn't like relative paths
    '';
  };
}
