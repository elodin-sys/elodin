{
  config,
  lib,
  pkgs,
  modulesPath,
  ...
}: let
  cfg = config.aleph.nvmeImage;
  inherit (config.aleph.fs) rootPartitionUUID;

  mkESPContent =
    pkgs.runCommand "mk-esp-contents" {
      nativeBuildInputs = with pkgs; [mypy python3];
    } ''
      install -m755 ${./mk-esp-contents.py} $out
      mypy \
        --no-implicit-optional \
        --disallow-untyped-calls \
        --disallow-untyped-defs \
        $out
    '';

  fdtPath = "${config.hardware.deviceTree.package}/${config.hardware.deviceTree.name}";

  espContents =
    pkgs.runCommand "aleph-esp-contents" {
      nativeBuildInputs = [pkgs.buildPackages.python3];
    } ''
      mkdir -p $out
      ${pkgs.buildPackages.python3}/bin/python3 ${mkESPContent} \
        --toplevel ${config.system.build.toplevel} \
        --output $out/ \
        --device-tree ${fdtPath}
    '';

  # 512 MiB FAT32 ESP, label BOOT (matches aleph-installer / fs.nix)
  espImage =
    pkgs.runCommand "aleph-esp.img" {
      nativeBuildInputs = with pkgs.buildPackages; [dosfstools mtools];
    } ''
      truncate -s 512M $out
      mkfs.vfat -F 32 -n BOOT $out
      # mtools needs directory trailing slashes when copying trees
      mcopy -i $out -s ${espContents}/* ::/
    '';

  rootImage = pkgs.callPackage "${modulesPath}/../lib/make-ext4-fs.nix" {
    storePaths = [config.system.build.toplevel];
    compressImage = false;
    volumeLabel = "APP";
    uuid = rootPartitionUUID;
    populateImageCommands = ''
      mkdir -p ./files
    '';
  };
in {
  options.aleph.nvmeImage = {
    enable = lib.mkEnableOption "Build ESP/root images for initrd NVMe flashing";
  };

  config = lib.mkIf cfg.enable {
    # Grow APP partition + filesystem to fill the NVMe on first boot
    boot.growPartition = true;
    fileSystems."/".autoResize = true;

    system.build.alephEspImage = espImage;
    system.build.alephRootImage = rootImage;
  };
}
