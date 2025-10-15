# Copyright 2022-2024 TII (SSRC) and the Ghaf contributors
# SPDX-License-Identifier: Apache-2.0
#
# Module which adds option ghaf.boot.loader.systemd-boot-dtb.enable
#
# By setting this option to true, device tree file gets copied to
# /boot-partition, and gets added to systemd-boot's entry.
#
{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.boot.loader.systemd-boot-dtb;
  inherit (lib) mkEnableOption mkIf;
in {
  options.boot.loader.systemd-boot-dtb = {
    enable = mkEnableOption "systemd-boot-dtb";
  };

  config = mkIf cfg.enable {
    boot.loader.systemd-boot = {
      extraFiles."dtbs/${config.hardware.deviceTree.name}" = "${config.hardware.deviceTree.package}/${config.hardware.deviceTree.name}";
      extraInstallCommands = ''
        # Find out the latest generation from loader.conf
        default_cfg=$(${pkgs.coreutils}/bin/cat /boot/loader/loader.conf | ${pkgs.gnugrep}/bin/grep default | ${pkgs.gawk}/bin/awk '{print $2}')
        FILEHASH=$(${pkgs.coreutils}/bin/sha256sum "${config.hardware.deviceTree.package}/${config.hardware.deviceTree.name}" | ${pkgs.coreutils}/bin/cut -d ' ' -f 1)
        FILENAME="/dtbs/$FILEHASH.dtb"
        ${pkgs.coreutils}/bin/cp -fv "${config.hardware.deviceTree.package}/${config.hardware.deviceTree.name}" "/boot$FILENAME"
        echo "devicetree $FILENAME" >> /boot/loader/entries/$default_cfg
        echo "test" > /boot/loader/entries/foo
      '';
    };
  };
}
