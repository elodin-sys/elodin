# Copyright 2022-2024 TII (SSRC) and the Ghaf contributors
# SPDX-License-Identifier: Apache-2.0
#
# Module which configures sd-image to generate images to be used with NVIDIA
# Jetson Orin AGX/NX devices. Supposed to be imported from format-module.nix.
#
# Generates ESP partition contents mimicking systemd-boot installation. Can be
# used to generate both images to be used in flashing script, and image to be
# flashed to external disk. NVIDIA's edk2 does not seem to care to much about
# the partition types, as long as there is a FAT partition, which contains
# EFI-directory and proper kind of structure, it finds the EFI-applications and
# boots them successfully.
#
# source: https://github.com/tiiuae/ghaf/blob/main/modules/reference/hardware/jetpack/nvidia-jetson-orin/sdimage.nix
{
  config,
  pkgs,
  modulesPath,
  lib,
  ...
}: {
  imports = [(modulesPath + "/installer/sd-card/sd-image.nix")];

  sdImage = let
    mkESPContentSource = pkgs.substituteAll {
      src = ./mk-esp-contents.py;
      isExecutable = true;
      inherit (pkgs.buildPackages) python3;
    };
    mkESPContent =
      pkgs.runCommand "mk-esp-contents"
      {
        nativeBuildInputs = with pkgs; [
          mypy
          python3
        ];
      }
      ''
        install -m755 ${mkESPContentSource} $out
        mypy \
          --no-implicit-optional \
          --disallow-untyped-calls \
          --disallow-untyped-defs \
          $out
      '';
    fdtPath = "${config.hardware.deviceTree.package}/${config.hardware.deviceTree.name}";
  in {
    firmwareSize = 256;
    populateFirmwareCommands = ''
      mkdir -pv firmware
      ${mkESPContent} \
        --toplevel ${config.system.build.toplevel} \
        --output firmware/ \
        --device-tree ${fdtPath}
    '';
    populateRootCommands = '''';
    postBuildCommands = ''
      fdisk_output=$(fdisk -l "$img")

      # Offsets and sizes are in 512 byte sectors
      blocksize=512

      # ESP partition offset and sector count
      part_esp=$(echo -n "$fdisk_output" | tail -n 2 | head -n 1 | tr -s ' ')
      part_esp_begin=$(echo -n "$part_esp" | cut -d ' ' -f2)
      part_esp_count=$(echo -n "$part_esp" | cut -d ' ' -f4)

      # root-partition offset and sector count
      part_root=$(echo -n "$fdisk_output" | tail -n 1 | head -n 1 | tr -s ' ')
      part_root_begin=$(echo -n "$part_root" | cut -d ' ' -f3)
      part_root_count=$(echo -n "$part_root" | cut -d ' ' -f4)

      echo -n $part_esp_begin > $out/esp.offset
      echo -n $part_esp_count > $out/esp.size
      echo -n $part_root_begin > $out/root.offset
      echo -n $part_root_count > $out/root.size
    '';
  };

  #TODO: should we use the default
  #https://github.com/NixOS/nixpkgs/blob/master/nixos/modules/installer/sd-card/sd-image.nix#L177
  fileSystems."/boot" = {
    device = "/dev/disk/by-label/${config.sdImage.firmwarePartitionName}";
    fsType = "vfat";
  };
}
