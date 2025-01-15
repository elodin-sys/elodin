{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
    jetpack.url = "github:elodin-sys/jetpack-nixos/nvpmode-fix";
    jetpack.inputs.nixpkgs.follows = "nixpkgs";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    #jetpack.url = "github:anduril/jetpack-nixos/master";
    flake-utils.url = "github:numtide/flake-utils";
    crane = {
      url = "github:ipetkov/crane";
    };
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
    jetpack,
    crane,
    rust-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [jetpack.overlays.default rust-overlay.overlays.default];
        };
      in rec {
        nixosModules.default = {
          pkgs,
          config,
          ...
        }: let
          file-bridge = ./aleph-file-bridge;
        in {
          imports = [
            "${nixpkgs}/nixos/modules/profiles/minimal.nix"
            jetpack.nixosModules.default
            ./modules/usb_eth.nix
            ./modules/hardware.nix
            ./modules/elodin_dev.nix
            ./modules/raw.nix
            ./modules/minimal.nix
          ];

          nixpkgs.overlays = [
            jetpack.overlays.default
            rust-overlay.overlays.default
          ];
          systemd.services.file-bridge = {
            wantedBy = ["multi-user.target"];
            after = ["network.target"];
            description = "start aleph-file-bridge";
            serviceConfig = {
              Type = "exec";
              User = "root";
              ExecStart = "${file-bridge}";
              KillSignal = "SIGINT";
              Environment = "RUST_LOG=debug";
            };
          };
          system.stateVersion = "24.05";
          i18n.supportedLocales = [(config.i18n.defaultLocale + "/UTF-8")];
          services.openssh.settings.PasswordAuthentication = true;
          services.openssh.enable = true;
          services.openssh.settings.PermitRootLogin = "yes";
          security.sudo.wheelNeedsPassword = false;
          users.users.root.password = "root";
          networking.hostName = "aleph";
          environment.systemPackages = with pkgs; [
            pciutils
            usbutils
            nvme-cli
            vim
            htop
            dtc
          ];
        };
        networking.hostName = "aleph";
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
          default = cross-config [nixosModules.default];
        };
        packages = {
          default = nixosConfigurations.default.config.system.build.sdImage;
          ext4 = nixosConfigurations.default.config.system.build.raw;
          toplevel = nixosConfigurations.default.config.system.build.toplevel;
          initrd_flash = pkgs.writeShellScriptBin "initrd-flash-aleph" ''
            set -euo pipefail
            WORKDIR=$(mktemp -d)
            function on_exit() {
               echo "cleaning up workdir"
               rm -rf "$WORKDIR"
            }
            trap on_exit EXIT
            cp -r ${./flash-dance.tar.xz} $WORKDIR/flash_dance.tar.xz
            chmod -R u+w "$WORKDIR"
            export PATH=${pkgs.lib.makeBinPath pkgs.nvidia-jetpack.flash-tools.flashDeps}:${pkgs.dtc}/bin:$PATH
            cd "$WORKDIR"
            tar xf flash_dance.tar.xz
            cd "flash-dance"
            cp ${packages.ext4} rootfs.ext4
            sudo ./initrd-flash
          '';
        };
      }
    );
}
