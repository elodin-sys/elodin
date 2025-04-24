{
  nixConfig = {
    extra-substituters = ["http://ci-arm1.elodin.dev:5000"];
    extra-trusted-public-keys = [
      "builder-cache-1:q7rDGIQgkg1nsxNEg7mHN1kEDuxPmJhQpuIXCCwLj8E="
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
    overlay = final: prev:
      (prev.lib.packagesFromDirectoryRecursive {
        directory = ./pkgs;
        callPackage = path: args: final.callPackage path (args // {inherit crane;});
      })
      // (rust-overlay.overlays.default final prev)
      // {
        inherit (final.nvidia-jetpack) cudaPackages;
        opencv4 = prev.opencv4.override {inherit (final) cudaPackages;};
      };
    modules = {
      usb-eth = ./modules/usb-eth.nix;
      hardware = ./modules/hardware.nix;
      minimal = ./modules/minimal.nix;
      sd-image = ./modules/sd-image.nix;
      aleph-dev = ./modules/aleph-dev.nix;
      elodin-db = ./modules/elodin-db.nix;
      aleph-serial-bridge = ./modules/aleph-serial-bridge.nix;
      tegrastats-bridge = ./modules/tegrastats-bridge.nix;
      mekf = ./modules/mekf.nix;
      aleph-setup = ./modules/aleph-setup.nix;
      wifi = ./modules/wifi.nix;
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
        ]
        ++ builtins.attrValues modules;
      nixpkgs.overlays = [
        jetpack.overlays.default
        rust-overlay.overlays.default
        overlay
        (final: prev: {
          inherit (final.nvidia-jetpack) cudaPackages;
          opencv4 = prev.opencv4.override {inherit (final) cudaPackages;};
        })
      ];
      system.stateVersion = "24.11";
      i18n.supportedLocales = [(config.i18n.defaultLocale + "/UTF-8")];
      services.openssh.settings.PasswordAuthentication = true;
      services.openssh.enable = true;
      services.openssh.settings.PermitRootLogin = "yes";
      # services.nvpmodel.enable = false;
      # services.nvfancontrol.enable = false;
      # Disable all the power saving features. They all negatively impact reliability.
      security.sudo.wheelNeedsPassword = false;
      users.users.root.password = "root";
      environment.etc."elodin-version" = let
        rustToolchain = p: p.rust-bin.stable."1.85.0".default;
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
      in {
        text = (craneLib.crateNameFromCargoToml {cargoToml = ../../Cargo.toml;}).version;
        enable = true;
      };
    };
    installerSystem = module: let
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
    in {
      default = defaultNixosConfig;
      installer = installerNixosConfig;
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
        apps.deploy = {
          type = "app";
          program = "${pkgs.writeScript "deploy" (builtins.readFile ./deploy.sh)}";
        };
      }
    )
    // rec {
      nixosModules =
        {
          default = defaultModule;
          jetpack = jetpack.nixosModules.default;
        }
        // modules;
      overlays.default = overlay;
      overlays.jetpack = jetpack.overlays.default;
      nixosConfigurations = installerSystem nixosModules.default;
      packages.aarch64-linux = {
        default = nixosConfigurations.default.config.system.build.sdImage;
        toplevel = nixosConfigurations.default.config.system.build.toplevel;
        sdimage = nixosConfigurations.installer.config.system.build.sdImage;
      };
      packages.x86_64-linux = {
        flash-uefi = nixpkgs.legacyPackages.x86_64-linux.runCommand "flash-uefi" {} ''
          mkdir -p $out
          cp ${jetpack.outputs.packages.x86_64-linux.flash-orin-nx-devkit}/bin/flash-orin-nx-devkit $out/flash-uefi
          sed -i '46i\cp ${./tegra234-mb2-bct-misc-p3767-0000.dts} bootloader/t186ref/BCT/tegra234-mb2-bct-misc-p3767-0000.dts' $out/flash-uefi
          chmod +x $out/flash-uefi
        '';
      };
      lib.installerSystem = installerSystem;
      templates.default = {
        path = ./template;
        description = "custom nixos image";
      };
    };
}
