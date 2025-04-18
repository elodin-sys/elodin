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
  };
  outputs = {
    self,
    nixpkgs,
    nixpkgs-unstable,
    flake-utils,
    jetpack,
    crane,
    rust-overlay,
  }: let
    system = "aarch64-linux";
    pkgs = import nixpkgs {inherit system;};
    defaultModule = {
      pkgs,
      config,
      lib,
      ...
    }: {
      imports = [
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
        ./modules/tegrastats-bridge.nix
        ./modules/mekf.nix
        ./modules/aleph-setup.nix
      ];

      nixpkgs.overlays = [
        jetpack.overlays.default
        rust-overlay.overlays.default
        (final: prev: {
          serial-bridge = final.callPackage ./pkgs/aleph-serial-bridge.nix {inherit crane;};
          elodin-db = final.callPackage ./pkgs/elodin-db.nix {inherit crane;};
          mekf = final.callPackage ./pkgs/mekf.nix {inherit crane;};
          aleph-status = final.callPackage ./pkgs/aleph-status.nix {inherit crane;};
          aleph-setup = final.callPackage ./pkgs/aleph-setup.nix {inherit crane;};
          tegrastats-bridge = final.callPackage ./pkgs/tegrastats-bridge.nix {inherit crane;};
        })
      ];
      system.stateVersion = "24.05";
      i18n.supportedLocales = [(config.i18n.defaultLocale + "/UTF-8")];
      services.openssh.settings.PasswordAuthentication = true;
      services.openssh.enable = true;
      services.openssh.settings.PermitRootLogin = "yes";
      # services.nvpmodel.enable = false;
      # services.nvfancontrol.enable = false;
      # Disable all the power saving features. They all negatively impact reliability.
      boot.extraModprobeConfig = ''
        options iwlwifi power_save=0 uapsd_disable=1 d0i3_disable=1
        options iwlmvm power_scheme=1
      '';
      security.sudo.wheelNeedsPassword = false;
      users.users.root.password = "root";
      networking.hostName = lib.mkForce "";
      systemd.services.set-hostname = {
        description = "Set hostname from /etc/machine-id";
        after = ["network.target"];
        before = ["systemd-hostnamed.service"];
        serviceConfig = {
          User = "root";
          Group = "root";
          Type = "oneshot";
          RemainAfterExit = true;
        };
        script = ''
          ID_PREFIX=$(head -c 4 /etc/machine-id)
          echo "aleph-$ID_PREFIX" > /etc/hostname
          echo "Hostname set to aleph-$ID_PREFIX"
        '';
        wantedBy = ["multi-user.target"];
      };
      networking.dhcpcd.enable = true;
      networking.wireless.iwd = {
        enable = true;
        settings = {
          IPv6 = {
            Enabled = true;
          };
          Settings = {
            AutoConnect = true;
          };
        };
      };
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
      environment.etc."elodin-version" = let
        rustToolchain = p: p.rust-bin.stable."1.85.0".default;
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
      in {
        text = (craneLib.crateNameFromCargoToml {cargoToml = ../../Cargo.toml;}).version;
        enable = true;
      };
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
