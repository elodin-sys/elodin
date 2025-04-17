{
  nixConfig = {
    extra-substituters = ["http://ci-arm1.elodin.dev:5000"];
    extra-trusted-public-keys = [
      "builder-cache-1:rEmIQJ4ChX5bopj3to1Ow7McFb7kLnXIQJYqawVlKEs="
    ];
  };
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    jetpack.url = "github:anduril/jetpack-nixos/master";
    jetpack.inputs.nixpkgs.follows = "nixpkgs";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
    crane = {
      url = "github:ipetkov/crane";
    };
    deploy-rs = {
      url = "github:serokell/deploy-rs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.utils.follows = "flake-utils";
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
    deploy-rs,
  }: let
    system = "aarch64-linux";
    pkgs = import nixpkgs {inherit system;};
    deployPkgs = import nixpkgs {
      inherit system;
      overlays = [
        deploy-rs.overlays.default
        (self: super: {
          deploy-rs = {
            inherit (pkgs) deploy-rs;
            lib = super.deploy-rs.lib;
          };
        })
      ];
    };
    defaultModule = {
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
          ./modules/aleph-dev.nix
          ./modules/elodin-db.nix
          ./modules/aleph-serial-bridge.nix
          ./modules/mekf.nix
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
      services.nvpmodel.enable = false;
      services.nvfancontrol.enable = false;
      systemd.services."wifi-powersave@wlP1p1s0" = {
        description = "Disables power-save on WiFi interface wlP1p1s0";
        bindsTo = ["sys-subsystem-net-devices-wlP1p1s0.device"];
        after = ["sys-subsystem-net-devices-wlP1p1s0.device"];
        wantedBy = ["multi-user.target"];
        serviceConfig = {
          Type = "oneshot";
          ExecStart = "/run/current-system/sw/bin/iw wlP1p1s0 set power_save off";
          RemainAfterExit = true;
        };
      };
      security.sudo.wheelNeedsPassword = false;
      users.users.root.password = "root";
      networking.hostName = "aleph";
      networking.wireless.enable = true;
      networking.dhcpcd.enable = true;
      nix.settings.trusted-users = ["root" "@wheel"];
      nix.settings.experimental-features = ["nix-command" "flakes"];
      nix.settings.trusted-public-keys = [
        "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
        "builder-cache-1:rEmIQJ4ChX5bopj3to1Ow7McFb7kLnXIQJYqawVlKEs="
      ];
      nix.settings.extra-substituters = [
        "https://cache.nixos.org"
        "http://ci-arm1.elodin.dev:5000"
      ];
      security.pam.loginLimits = [
        {
          domain = "*";
          type = "soft";
          item = "nofile";
          value = "65536";
        }
        {
          domain = "*";
          type = "hard";
          item = "nofile";
          value = "524288";
        }
      ];
    };
    appModule = {lib, ...}: {
      imports = [defaultModule];
      fileSystems."/".device = lib.mkForce "/dev/disk/by-label/APP";
      fileSystems."/boot".device = lib.mkForce "/dev/disk/by-label/BOOT";
    };
    installerModule = {...}: {
      imports = [
        defaultModule
        ./modules/installer.nix
      ];
      aleph.installer.system = defaultNixosConfig.config.system.build.toplevel;
    };
    defaultNixosConfig = nixpkgs.lib.nixosSystem {
      inherit system;
      modules = [appModule];
    };
    installerNixosConfig = nixpkgs.lib.nixosSystem {
      inherit system;
      modules = [installerModule];
    };
  in
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells = {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              just
              deploy-rs.packages.${system}.deploy-rs
            ];
          };
        };
      }
    )
    // rec {
      nixosModules = {
        default = defaultModule;
        app = appModule;
        installer = installerModule;
      };
      nixosConfigurations = {
        default = defaultNixosConfig;
        installer = installerNixosConfig;
      };
      deploy.nodes.aleph = {
        hostname = "aleph.local";
        profiles.system = {
          user = "root";
          path = deployPkgs.deploy-rs.lib.activate.nixos self.nixosConfigurations.default;
        };
      };
      packages.aarch64-linux = {
        default = nixosConfigurations.default.config.system.build.sdImage;
        toplevel = nixosConfigurations.default.config.system.build.toplevel;
        sdimage = nixosConfigurations.installer.config.system.build.sdImage;
        flash-uefi = pkgs.runCommand "flash-uefi" {} ''
          mkdir -p $out
          cp ${jetpack.outputs.packages.aarch64-linux.flash-orin-nx-devkit}/bin/flash-orin-nx-devkit $out/flash-uefi
          sed -i '46i\cp ${./tegra234-mb2-bct-misc-p3767-0000.dts} bootloader/t186ref/BCT/tegra234-mb2-bct-misc-p3767-0000.dts' $out/flash-uefi
          chmod +x $out/flash-uefi
        '';
      };
    };
}
