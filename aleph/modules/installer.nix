{
  pkgs,
  config,
  lib,
  ...
}: let
  installer = pkgs.writeShellApplication {
    name = "aleph-installer";

    runtimeInputs = with pkgs; [
      parted
      gum
      e2fsprogs # mkfs.ext4
      dosfstools # mkfs.fat
      util-linux # mount
      coreutils # mkdir
      nixos-install-tools # nixos-install
    ];

    text = ''
      gum confirm "Install ℵ NixOS to /dev/nvme0n1"

      # format
      export GUM_SPIN_SHOW_OUTPUT=1
      # Clear any leftover partition table / filesystem signatures from a
      # previous attempt; otherwise mkfs stops on an interactive "proceed?"
      # prompt that can't be answered under `gum spin` on a serial console.
      wipefs -af /dev/nvme0n1
      gum spin --title "Creating GPT" -- parted -s /dev/nvme0n1 -- mklabel gpt
      gum spin --title "Creating ROOT" -- parted -s /dev/nvme0n1 -- mkpart root ext4 512MB 100%
      gum spin --title "Created ESP" -- parted -s /dev/nvme0n1 -- mkpart ESP fat32 1MB 512MB
      gum spin --title "Set ESP ON" -- parted -s /dev/nvme0n1 -- set 2 esp on
      # -F: force, so a residual ext4 superblock doesn't trigger a prompt.
      gum spin --title "Format ROOT" -- mkfs.ext4 -F -q -L APP -U ${config.aleph.fs.rootPartitionUUID} /dev/nvme0n1p1
      gum spin --title "Format ESP" -- mkfs.fat -F 32 -n BOOT /dev/nvme0n1p2

      # install
      mkdir -p /mnt
      mount /dev/nvme0n1p1 /mnt
      mkdir -p /mnt/boot
      mount -o umask=077 /dev/nvme0n1p2 /mnt/boot
      # The install target is already valid in the offline store (registered at
      # boot by `register-nix-paths`, see modules/ram-installer.nix).
      # nixos-install populates /mnt by *substituting* the closure from the local
      # store — it runs `nix-env --store /mnt --extra-substituters auto?trusted=1
      # --set <system>`. So we must NOT pass `--option substitute false`: that
      # disables the local `auto` substituter too, and the install fails with
      # "no substituter that can build it". Offline quiet is achieved instead by
      # dropping the network binary caches in ram-installer.nix, leaving only the
      # local `auto` substituter. Run nixos-install directly (not under
      # `gum spin`) so progress/errors are visible.
      echo "Installing ℵ NixOS (this can take a few minutes)..."
      nixos-install --system ${config.aleph.installer.system} --no-root-passwd
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
