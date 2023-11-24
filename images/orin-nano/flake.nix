{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    jetpack.url = "github:danielfullmer/jetpack-nixos/r35.4.1";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
    jetpack,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };
      in rec {
        nixosModules.default = {
          pkgs,
          lib,
          config,
          modulesPath,
          ...
        }: let
          rootfsImage = pkgs.callPackage "${modulesPath}/../lib/make-ext4-fs.nix" {
            storePaths = [config.system.build.toplevel];
            volumeLabel = "nixos";
            populateImageCommands = ''
              mkdir -p ./files/boot
              ${config.boot.loader.generic-extlinux-compatible.populateCmd} -c ${config.system.build.toplevel} -d ./files/boot
              sed -i 's/..\/nixos/\/boot\/nixos/g' ./files/boot/extlinux/extlinux.conf # Jetson doesn't like relative paths
            '';
          };
          efiFirmware = pkgs.stdenv.mkDerivation rec {
            pname = "l4tlauncher";
            version = "0.1.0";

            src = ./.;

            installPhase = ''
              mkdir -p $out
              cp -R BOOTAA64.efi $out/
            '';
          };
        in {
          imports = [
            "${nixpkgs}/nixos/modules/profiles/base.nix"
            #"${nixpkgs}/nixos/modules/installer/sd-card/sd-image.nix"
            jetpack.nixosModules.default
          ];
          boot.loader.grub.enable = false;
          boot.loader.generic-extlinux-compatible.enable = true;
          boot.loader.generic-extlinux-compatible.useGenerationDeviceTree = true;

          i18n.supportedLocales = [(config.i18n.defaultLocale + "/UTF-8")];
          fileSystems."/" = {
            device = "/dev/disk/by-label/nixos";
            fsType = "ext4";
            autoResize = true;
          };

          fileSystems."/boot" = {
            device = "/dev/disk/by-label/boot";
            fsType = "vfat";
          };

          system.build.sdImage = pkgs.callPackage ({
            stdenv,
            dosfstools,
            e2fsprogs,
            mtools,
            libfaketime,
            util-linux,
            zstd,
          }:
            stdenv.mkDerivation {
              name = "orin-nano-image";

              nativeBuildInputs = [dosfstools e2fsprogs libfaketime mtools util-linux];

              buildCommand = ''
                mkdir -p $out/nix-support $out/sd-image
                export img=$out/sd-image/orin-nano-image

                root_fs=${rootfsImage}
                echo $root_fs

                rootSizeBlocks=$(du -B 512 --apparent-size $root_fs | awk '{ print $1 }')
                imageSize=$((rootSizeBlocks * 512 + 1026048 * 512 + 33 * 512))
                echo "imageSize:"
                echo $imageSize
                echo $rootSizeBlocks
                truncate -s $imageSize $img

                sfdisk $img <<EOF
                  label: gpt
                  label-id: 13BFDDFB-8D0B-9E4F-978C-AC85BE1FCE85
                  unit: sectors
                  sector-size: 512

                  1 : start=        2048, size=     1024000, type=C12A7328-F81F-11D2-BA4B-00A0C93EC93B, uuid=869DFE3F-4B0F-7F41-9C70-D3934D969A22
                  2 : start=     1026048, size= $rootSizeBlocks, type=0FC63DAF-8483-4772-8E79-3D69D8477DE4, uuid=F8F48F84-9ADD-944A-A245-755CFF377FAA, name="APP"
                EOF

                # Copy the rootfs into the SD image
                eval $(partx $img -o START,SECTORS --nr 2 --pairs)
                dd conv=notrunc if=$root_fs of=$img seek=$START count=$SECTORS

                eval $(partx $img -o START,SECTORS --nr 1 --pairs)
                truncate -s $((SECTORS * 512)) firmware_part.img
                mkfs.vfat -F 32 --invariant  -n boot firmware_part.img
                mkdir firmware
                mkdir -p ./firmware/boot
                ${config.boot.loader.generic-extlinux-compatible.populateCmd} -c ${config.system.build.toplevel} -d ./firmware/boot
                sed -i 's/..\/nixos/\/boot\/nixos/g' ./firmware/boot/extlinux/extlinux.conf # Jetson doesn't like relative paths
                cat ./firmware/boot/extlinux/extlinux.conf
                mkdir -p ./firmware/EFI/BOOT
                cp ${efiFirmware}/* ./firmware/EFI/BOOT

                find firmware -exec touch --date=2000-01-01 {} +
                # Copy the populated /boot/firmware into the SD image
                cd firmware
                # Force a fixed order in mcopy for better determinism, and avoid file globbing
                for d in $(find . -type d -mindepth 1 | sort); do
                  faketime "2000-01-01 00:00:00" mmd -i ../firmware_part.img "::/$d"
                done
                for f in $(find . -type f | sort); do
                  mcopy -pvm -i ../firmware_part.img "$f" "::/$f"
                done
                cd ..
                fsck.vfat -vn firmware_part.img
                dd conv=notrunc if=firmware_part.img of=$img seek=$START count=$SECTORS



              '';
            }) {};

          # Avoids a bunch of extra modules we don't have in the tegra_defconfig, like "ata_piix",
          disabledModules = ["profiles/all-hardware.nix"];
          hardware.deviceTree.name = "tegra234-p3767-0003-p3509-a02.dtb";
          hardware.nvidia-jetpack = {
            enable = true;
            som = "orin-nano";
            carrierBoard = "devkit";
            #kernel.realtime = true;
          };
          hardware.firmware = [pkgs.linux-firmware];
          services.openssh.passwordAuthentication = true;
          services.openssh.enable = true;
          security.sudo.wheelNeedsPassword = false;
          users.users.root.password = "root";
          networking.wireless.enable = true;
          networking.firewall.enable = false;
          networking.dhcpcd.enable = true;
          users.users."sphw" = {
            openssh.authorizedKeys.keys = [
              "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMFaAHTd2YJXqJPwbaD8Z2NmB4oYvz3Rusdlh374g6Qw"
            ];
            isNormalUser = true;
            extraGroups = ["wheel"];
            initialPassword = "nixos";
          };
          environment.systemPackages = with pkgs; [
            parted
            busybox
            pciutils
            usbutils
            nvme-cli
            vim
            htop
          ];
        };
        nixosConfigurations = let
          toGuest = builtins.replaceStrings ["darwin"] ["linux"];
          cross-config = modules:
            if toGuest system == "aarch64-linux"
            then
              nixpkgs.lib.nixosSystem {
                system = "aarch64-linux";
                inherit modules;
              }
            else
              pkgs.pkgsCross.aarch64-multiplatform.nixos {
                imports =
                  modules;
              };
        in {
          default = cross-config [nixosModules.default];
        };
        packages = {
          default = nixosConfigurations.default.config.system.build.sdImage;
        };
      }
    );
}
