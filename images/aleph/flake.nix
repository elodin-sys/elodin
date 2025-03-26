{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    jetpack.url = "github:anduril/jetpack-nixos/master";
    jetpack.inputs.nixpkgs.follows = "nixpkgs";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
    };
    flake-utils.url = "github:numtide/flake-utils";
    crane = {
      url = "github:ipetkov/crane";
    };
  };
  outputs = {
    self,
    nixpkgs,
    nixpkgs-unstable,
    flake-utils,
    jetpack,
    crane,
    rust-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            jetpack.overlays.default
            rust-overlay.overlays.default
          ];
        };
      in rec {
        nixosModules.default = {
          pkgs,
          config,
          lib,
          ...
        }: {
          imports =
            [
              "${nixpkgs}/nixos/modules/profiles/minimal.nix"
              jetpack.nixosModules.default
              ./modules/usb-eth.nix
              ./modules/hardware.nix
              ./modules/minimal.nix
              ./modules/sd-image.nix
              ./modules/systemd-boot-dtb.nix
              ./modules/elodin-db.nix
              ./modules/aleph-serial-bridge.nix
              ./modules/mekf.nix
              ./modules/aleph-dev.nix
            ]
            ++ lib.optional (builtins.pathExists ./modules/elodin-dev.nix) ./modules/elodin-dev.nix;

          nixpkgs.overlays = [
            jetpack.overlays.default
            rust-overlay.overlays.default
            (final: prev: {
              serial-bridge = final.callPackage ./pkgs/aleph-serial-bridge.nix {inherit crane;};
              elodin-db = final.callPackage ./pkgs/elodin-db.nix {inherit crane;};
              mekf = final.callPackage ./pkgs/mekf.nix {inherit crane;};
            })
          ];
          system.stateVersion = "24.05";
          i18n.supportedLocales = [(config.i18n.defaultLocale + "/UTF-8")];
          services.openssh.settings.PasswordAuthentication = true;
          services.openssh.enable = true;
          services.openssh.settings.PermitRootLogin = "yes";
          security.sudo.wheelNeedsPassword = false;
          users.users.root.password = "root";
          networking.hostName = "aleph";
          networking.wireless.enable = true;
          networking.dhcpcd.enable = true;
          nix.settings.trusted-users = ["root" "@wheel"];
          environment.systemPackages = with pkgs; [
            pciutils
            usbutils
            nvme-cli
            vim
            htop
            dtc
          ];
        };
        nixosModules.app = {lib, ...}: {
          imports = [nixosModules.default];
          fileSystems."/".device = lib.mkForce "/dev/disk/by-label/APP";
          fileSystems."/boot".device = lib.mkForce "/dev/disk/by-label/BOOT";
        };
        nixosModules.installer = {...}: {
          imports = [
            nixosModules.default
            ./modules/installer.nix
          ];
          aleph.installer.system = nixosConfigurations.default.config.system.build.toplevel;
        };
        nixosConfigurations = let
          toGuest = builtins.replaceStrings ["darwin"] ["linux"];
          # this fun little hack lets you build nixos modules on macOS or Linux on either x86 or arm64
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
          default = cross-config [nixosModules.app];
          installer = cross-config [nixosModules.installer];
        };
        packages = {
          default = nixosConfigurations.default.config.system.build.sdImage;
          toplevel = nixosConfigurations.default.config.system.build.toplevel;
          sdimage = nixosConfigurations.installer.config.system.build.sdImage;
          flash-uefi = pkgs.runCommand "flash-uefi" {} ''
            mkdir -p $out
            cp ${jetpack.outputs.packages.${system}.flash-orin-nx-devkit}/bin/flash-orin-nx-devkit $out/flash-uefi
            sed -i '46i\cp ${./tegra234-mb2-bct-misc-p3767-0000.dts} bootloader/t186ref/BCT/tegra234-mb2-bct-misc-p3767-0000.dts' $out/flash-uefi
            chmod +x $out/flash-uefi
          '';
        };
      }
    );
}
