{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    jetpack.url = "github:anduril/jetpack-nixos/2c98c9d6c326d67ae5f4909db61238d31352e18c";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";

    jetpack.inputs.nixpkgs.follows = "nixpkgs";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs = {
    self,
    nixpkgs,
    nixpkgs-unstable,
    flake-utils,
    jetpack,
    rust-overlay,
  }: let
    system = "aarch64-linux";
    rustToolchain = p: p.rust-bin.fromRustupToolchainFile ../rust-toolchain.toml;
    gitJSONOverlay = builtins.fromJSON (builtins.readFile ./gitrepos.json);
    gitReposOverlay = final: prev: {
      cudaPackages = prev.cudaPackages_12; # CUDA 12 for JP6
      nvidia-jetpack = prev.nvidia-jetpack6.overrideScope (jetpackFinal: jetpackPrev: {
        gitRepos =
          jetpackPrev.gitRepos
          // (final.lib.mapAttrs (_: info:
            final.fetchgit {
              inherit (info) url rev sha256;
            })
          gitJSONOverlay);
      });
    };
    overlay = final: prev:
      (prev.lib.packagesFromDirectoryRecursive {
        directory = ./pkgs;
        callPackage = path: args: final.callPackage path (args // {inherit rustToolchain;});
      })
      // (rust-overlay.overlays.default final prev)
      // (gitReposOverlay final prev);

    baseModules = {
      default = defaultModule;
      jetpack = jetpack.nixosModules.default;
      usb-eth = ./modules/usb-eth.nix;
      hardware = ./modules/hardware.nix;
      minimal = ./modules/minimal.nix;
      fs = ./modules/fs.nix;
      aleph-base = ./modules/aleph-base.nix;
      aleph-setup = ./modules/aleph-setup.nix;
      wifi = ./modules/wifi.nix;
    };
    fswModules = {
      elodin-db = ./modules/elodin-db.nix;
      aleph-serial-bridge = ./modules/aleph-serial-bridge.nix;
      tegrastats-bridge = ./modules/tegrastats-bridge.nix;
      mekf = ./modules/mekf.nix;
      msp-osd = ./modules/msp-osd.nix;
      udp-component-broadcast = ./modules/udp-component-broadcast.nix;
      udp-component-receive = ./modules/udp-component-receive.nix;
      elodinsink = ./modules/elodinsink.nix;
    };
    devModules = {
      aleph-dev = ./modules/aleph-dev.nix;
    };
    defaultModule = {config, ...}: {
      imports = [
        "${nixpkgs}/nixos/modules/profiles/minimal.nix"
      ];
      nixpkgs.overlays = [
        jetpack.overlays.default
        overlay
      ];
      system.stateVersion = "25.05";
      i18n.supportedLocales = [(config.i18n.defaultLocale + "/UTF-8")];
      services.openssh.settings.PasswordAuthentication = true;
      services.openssh.enable = true;
      services.openssh.settings.PermitRootLogin = "yes";
      # services.nvpmodel.enable = false;
      # services.nvfancontrol.enable = false;
      security.sudo.wheelNeedsPassword = false;
      users.users.root.password = "root";
    };
    installerSystem = module: let
      baseNixosConfig = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [module];
      };
    in
      nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          module
          ({...}: {
            imports = [./modules/installer.nix];
            aleph.sd.enable = true;
            aleph.installer.system = baseNixosConfig.config.system.build.toplevel;
          })
        ];
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
              zstd
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
      nixosModules = baseModules // fswModules // devModules;
      overlays.default = overlay;
      overlays.jetpack = jetpack.overlays.default;
      overlays.gitRepos = gitReposOverlay;
      nixosConfigurations = {
        default = nixpkgs.lib.nixosSystem {
          inherit system;
          modules =
            builtins.attrValues baseModules
            ++ builtins.attrValues fswModules
            ++ builtins.attrValues devModules;
        };
        installer = installerSystem ({...}: {
          imports = builtins.attrValues baseModules;
        });
      };
      packages.aarch64-linux = {
        toplevel = nixosConfigurations.default.config.system.build.toplevel;
        sdimage = nixosConfigurations.installer.config.system.build.sdImage;
      };
      packages.x86_64-linux = let
        flash-cross = jetpack.nixosConfigurations."orin-nx-devkit".extendModules {
          modules = [
            {nixpkgs.buildPlatform.system = "x86_64-linux";}
          ];
        };
      in {
        flash-uefi = nixpkgs.legacyPackages.x86_64-linux.runCommand "flash-uefi" {} ''
          mkdir -p $out
          cp ${flash-cross.config.system.build.legacyFlashScript}/bin/flash-orin-nx-devkit $out/flash-uefi
          sed -i '46i\cp ${./tegra234-mb2-bct-misc-p3767-0000.dts} bootloader/generic/BCT/tegra234-mb2-bct-misc-p3767-0000.dts' $out/flash-uefi
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
