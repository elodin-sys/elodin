{
  nixConfig = {
    extra-substituters = ["https://elodin-nix-cache.s3.us-west-2.amazonaws.com"];
    extra-trusted-public-keys = [
      "elodin-cache-1:vvbmIQvTOjcBjIs8Ri7xlT2I3XAmeJyF5mNlWB+fIwM="
    ];
    fallback = true;
  };
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    jetpack.url = "github:anduril/jetpack-nixos";
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
    gitReposOverlay = final: prev: {
      cudaPackages = prev.cudaPackages_12; # CUDA 12 for JP6
      nvidia-jetpack = prev.nvidia-jetpack6.overrideScope (jetpackFinal: jetpackPrev: {
        gitRepos =
          jetpackPrev.gitRepos
          // {
            "kernel/kernel-jammy-src" = final.fetchgit {
              url = "https://github.com/elodin-sys/aleph-orin-baseboard-kernel.git";
              rev = "4985987d7398523e967b354b07cd71451825bc7a";
              sha256 = "sha256-Wev5EONyEhSQxVMtwX3UhdvIs/QQX295ipXpnLrEtW0=";
            };
            "hardware/nvidia/t23x/nv-public" = final.fetchgit {
              url = "https://github.com/elodin-sys/aleph-jetson-orin-baseboard-nvidia-t23x-public-dts.git";
              rev = "c2ec44d2441400563b9a97babf212b7682bcd0aa";
              sha256 = "sha256-FDw1btoKCJ8OpeBYz1FLGH5+6Ry4Y4woshi8aDEw1gI=";
            };
            "hardware/nvidia/tegra/nv-public" = final.fetchgit {
              url = "https://github.com/OE4T/tegra-public-dts.git";
              rev = "8ba5d53ef1e1753f9f2a5b1f7b7b5fc5039de68e";
              sha256 = "sha256-NMp7UY0OlH2ddBSrUzCUSLkvnWrELhz8xH/dkV86ids=";
            };
            "hwpm" = final.fetchgit {
              url = "https://github.com/OE4T/linux-hwpm.git";
              rev = "d47dc62f4011ffbb0353ba43df5cfb42b967bef2";
              sha256 = "sha256-otOVFeF+8XKORWMXTRTcXQUXvojdwInVC3jPXTgrk3A=";
            };
            "kernel-devicetree" = final.fetchgit {
              url = "https://github.com/OE4T/kernel-devicetree.git";
              rev = "19952c8e25702e9de23500c3b1fb351bf4380446";
              sha256 = "sha256-mFmxO7rg1DWnYK+HDFQnc9XLpS4lwXfSXGOibKC4FPY=";
            };
            "nvdisplay" = final.fetchgit {
              url = "https://github.com/OE4T/nv-kernel-display-driver.git";
              rev =  "2c448bfbebf4314293fd432d2a0b33f66e5a4815";
              sha256 =  "0adq1wvnv36z2dpl39ixban7wjbzy623s9v0crdyaj3s1f4fi642";
            };
            "nvgpu" = final.fetchgit {
              url = "https://github.com/OE4T/linux-nvgpu.git";
              rev = "21d928824dc7ca3dc17603a53b11edc6641ace2d";
              sha256 = "sha256-5OIjSK1grV/nz3hHkeZKRb7dXIQAokNcChGCtzlfBKs=";
            };
            "nvidia-oot" = final.fetchgit {
              url = "https://nv-tegra.nvidia.com/linux-nv-oot.git";
              rev = "efa698bed82f27e403537b7ecf82743e696d17ef";
              sha256 ="1x5czgiclfkgi999rld7dg57qffw1qjhawvyj1hjkmqrdr2r4nb3";
            };
          };
      });
    };
    overlay = final: prev:
      (prev.lib.packagesFromDirectoryRecursive {
        directory = ./pkgs;
        callPackage = path: args: final.callPackage path (args // {inherit rustToolchain;});
      })
      // {
        memserve = final.callPackage ../nix/pkgs/memserve.nix {inherit rustToolchain;};
      }
      // (rust-overlay.overlays.default final prev)
      // (gitReposOverlay final prev);
    baseModules = {
      default = defaultModule;
      jetpack = jetpack.nixosModules.default;
      usb-eth = ./modules/usb-eth.nix;
      hardware = ./modules/hardware-515.nix;
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
    };
    devModules = {
      # Temporarily disabled for nixpkgs 25.05 compatibility (CUDA issues)
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
      services.openssh.enable = true;
      services.openssh.settings = {
        KbdInteractiveAuthentication = true;
        PasswordAuthentication = true;
        PubkeyAuthentication = true;
        PermitRootLogin = "yes";
      };
      security.sudo.wheelNeedsPassword = false;
      security.pam.services.sshd.text = ''
        auth        sufficient  pam_permit.so
        account     sufficient  pam_permit.so
        password    sufficient  pam_permit.so
        session     sufficient  pam_permit.so
      '';
      users.users.root.password = "root";
      # services.nvpmodel.enable = false;
      # services.nvfancontrol.enable = false;
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
      packages.x86_64-linux = {
        flash-uefi = nixpkgs.legacyPackages.x86_64-linux.runCommand "flash-uefi" {} ''
          mkdir -p $out
          cp ${jetpack.outputs.packages.x86_64-linux.flash-orin-nx-devkit}/bin/flash-orin-nx-devkit-cross $out/flash-uefi
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
