{
  pkgs,
  config,
  lib,
  ...
}: let
  installer = pkgs.writeShellApplication {
    name = "aleph-installer";

    runtimeInputs = with pkgs; [parted gum];

    text = ''
      gum confirm "Install ℵ NixOS to /dev/nvme0n1"

      # format
      export GUM_SPIN_SHOW_OUTPUT=1
      gum spin --title "Creating GPT" -- parted -s /dev/nvme0n1 -- mklabel gpt
      gum spin --title "Creating ROOT" -- parted -s /dev/nvme0n1 -- mkpart root ext4 512MB 100%
      gum spin --title "Created ESP" -- parted -s /dev/nvme0n1 -- mkpart ESP fat32 1MB 512MB
      gum spin --title "Set ESP ON" -- parted -s /dev/nvme0n1 -- set 2 esp on
      gum spin --title "Format ROOT" -- mkfs.ext4 -q -L APP /dev/nvme0n1p1
      gum spin --title "Format ESP" -- mkfs.fat -F 32 -n BOOT /dev/nvme0n1p2

      # install
      mkdir -p /mnt
      mount /dev/nvme0n1p1 /mnt
      mkdir -p /mnt/boot
      mount -o umask=077 /dev/nvme0n1p2 /mnt/boot
      export GUM_SPIN_SHOW_OUTPUT=0
      gum spin --title "Installing ℵ NixOS" -- nixos-install --system ${config.aleph.installer.system} --no-root-passwd
      echo "## Installed ℵ NixOS!" | gum format
      echo ":tada::tada::tada:" | gum format -t emoji
    '';
  };
in {
  options.aleph.installer = {
    system = lib.mkOption {
      type = lib.types.package;
    };
  };
  config.nixpkgs.overlays = [
    (
      final: prev: {
        gum = prev.gum.overrideAttrs {
          postInstall = ""; # gum's postInstall script is broken on cross compile
        };
      }
    )
  ];
  config.environment.systemPackages = [installer];
}
