{
  nixConfig = {
    extra-substituters = ["https://elodin-nix-cache.s3.us-west-2.amazonaws.com"];
    extra-trusted-public-keys = [
      "elodin-cache-1:vvbmIQvTOjcBjIs8Ri7xlT2I3XAmeJyF5mNlWB+fIwM="
    ];
    fallback = true;
  };
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    jetpack.url = "github:anduril/jetpack-nixos/4a4e93a7b3fbe1915870ec54002c616f01367195";
    rust-overlay.url = "github:oxalica/rust-overlay/77a8263847fb02dc49dbe377278ef6b952f1c6bb";

    jetpack.inputs.nixpkgs.follows = "nixpkgs";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs = {
    self,
    nixpkgs,
    nixpkgs-unstable,
    jetpack,
    rust-overlay,
  }: let
    # Separate Aleph's fixed NixOS target from host-scoped flake outputs
    alephSystem = "aarch64-linux";
    supportedSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forAllSystems = f:
    # Equivalent to using flake-utils
      nixpkgs.lib.genAttrs supportedSystems (system: f nixpkgs.legacyPackages.${system});

    rustToolchain = p: p.rust-bin.fromRustupToolchainFile ../rust-toolchain.toml;
    gitJSONOverlay = builtins.fromJSON (builtins.readFile ./gitrepos.json);
    secureBzip2Overlay = _final: prev: {
      # JetPack only needs a non-ancient unpacker; avoid nixpkgs' insecure 1.1 snapshot.
      bzip2_1_1 = prev.bzip2;
    };
    gitReposOverlay = final: prev: {
      # Our packing of deepstream7 still needs nvidia sources, so we can't use upstream nixpkgs yet!
      # Start with upstream nixpkgs _cuda/manifests and overlay jetpack ones (where prevCuda is jetpack-nixos cuda)
      _cuda = prev._cuda.extend (_: prevCuda: {
        manifests =
          final.lib.recursiveUpdate
          (import "${nixpkgs}/pkgs/development/cuda-modules/_cuda/manifests" {inherit (final) lib;})
          prevCuda.manifests;
      });
      # Override jetpack-nixos gitrepos with other sources specific to the aleph carrier board configuration
      nvidia-jetpack6 = prev.nvidia-jetpack6.overrideScope (jetpackFinal: jetpackPrev: {
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
      // (secureBzip2Overlay final prev)
      // (gitReposOverlay final prev)
      // {
        # cudaSupport (aleph-cuda.nix) pulls jax-cuda12-plugin -> nccl (badPlatforms
        # aarch64), unused (sim=cranelift, render=Vulkan). Force CPU jax everywhere.
        pythonPackagesExtensions =
          (prev.pythonPackagesExtensions or [])
          ++ [(_pyfinal: pyprev: {jax = pyprev.jax.override {cudaSupport = false;};})];
      };

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
      elodin = ./modules/elodin.nix;
      elodin-db = ./modules/elodin-db.nix;
      aleph-serial-bridge = ./modules/aleph-serial-bridge.nix;
      tegrastats-bridge = ./modules/tegrastats-bridge.nix;
      mekf = ./modules/mekf.nix;
      msp-osd = ./modules/msp-osd.nix;
      udp-component-broadcast = ./modules/udp-component-broadcast.nix;
      udp-component-receive = ./modules/udp-component-receive.nix;
      elodinsink = ./modules/elodinsink.nix;
    };
    stmModules = {
      sensor-fw = ./modules/sensor-fw.nix;
      c-blinky = ./modules/c-blinky.nix;
    };
    stmConfigurationModules = {
      stm = ./modules/stm.nix;
    };
    devModules = {
      aleph-cuda = ./modules/aleph-cuda.nix;
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

    # A set of presets that flash different programs onto the expansion board STM
    configurationPresets = import ./modules/custom-configurations.nix;

    installerSystem = module: let
      baseNixosConfig = nixpkgs.lib.nixosSystem {
        system = alephSystem;
        modules = [module];
      };
    in
      nixpkgs.lib.nixosSystem {
        system = alephSystem;
        modules = [
          module
          ({...}: {
            imports = [./modules/installer.nix];
            aleph.sd.enable = true;
            aleph.installer.system = baseNixosConfig.config.system.build.toplevel;
          })
        ];
      };
    baseConfigurationModules =
      builtins.attrValues baseModules
      ++ builtins.attrValues fswModules
      ++ builtins.attrValues devModules;

    # Create a NixOS system configuration from the shared base modules plus any extra modules.
    mkConfiguration = extraModules:
      nixpkgs.lib.nixosSystem {
        system = alephSystem;
        modules = baseConfigurationModules ++ extraModules;
      };
    customConfigurations = {
      base = mkConfiguration [];
      c-blinky = mkConfiguration [configurationPresets.preset-c-blinky];
      sensor-fw = mkConfiguration [configurationPresets.preset-sensor-fw];
      m10q = mkConfiguration [configurationPresets.preset-m10q];
      m9n = mkConfiguration [configurationPresets.preset-m9n];
    };
  in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          buildInputs = with pkgs; [
            just
            zstd
          ];
        };
      });
      apps = let
        baseApps = forAllSystems (pkgs: {
          deploy = {
            type = "app";
            program = "${pkgs.writeScript "deploy" (builtins.readFile ./deploy.sh)}";
          };
        });
      in
        baseApps
        // {
          x86_64-linux =
            baseApps.x86_64-linux
            // {
              flash-initrd = {
                type = "app";
                program = "${self.packages.x86_64-linux.flash-initrd}/flash-initrd";
              };
            };
        };
    }
    // rec {
      nixosModules = baseModules // fswModules // stmModules // stmConfigurationModules // devModules // configurationPresets;
      overlays.default = overlay;
      overlays.jetpack = jetpack.overlays.default;
      overlays.gitRepos = gitReposOverlay;
      nixosConfigurations =
        {
          # .#nixosConfigurations.default.config.system.build.toplevel will apply the
          # sensor-fw sketch onto the expansion board STM by default
          default = mkConfiguration [configurationPresets.preset-sensor-fw];
          installer = installerSystem ({...}: {
            imports = builtins.attrValues baseModules;
          });
        }
        // customConfigurations;
      packages.aarch64-linux = {
        toplevel = nixosConfigurations.default.config.system.build.toplevel;
        sdimage = nixosConfigurations.installer.config.system.build.sdImage;
      };
      packages.x86_64-linux = let
        # baseModules only (no FSW/CUDA) — flash tooling + NVMe install image.
        flashMinimalModules = [
          ({...}: {imports = builtins.attrValues baseModules;})
        ];
        flashInstallSystem = nixpkgs.lib.nixosSystem {
          system = alephSystem;
          modules =
            flashMinimalModules
            ++ [{aleph.nvmeImage.enable = true;}];
        };
        flashToolSystem = nixpkgs.lib.nixosSystem {
          system = alephSystem;
          modules = flashMinimalModules;
        };
        espImage = flashInstallSystem.config.system.build.alephEspImage;
        rootImage = flashInstallSystem.config.system.build.alephRootImage;
        rootPartitionUUID = flashInstallSystem.config.aleph.fs.rootPartitionUUID;

        flash-cross = jetpack.nixosConfigurations."orin-nx-devkit".extendModules {
          modules = [
            {nixpkgs.buildPlatform.system = "x86_64-linux";}
            {nixpkgs.overlays = [secureBzip2Overlay];}
          ];
        };

        # Explicit list: nixpkgs defaults (e.g. tpm-tis) are missing from tegra_defconfig
        # and break makeModulesClosure (allowMissing = false).
        flashInitrdModules = [
          "mtdblock"
          "spi_tegra210_quad"
          "libcomposite"
          "udc-core"
          "tegra-xudc"
          "xhci-tegra"
          "u_serial"
          "usb_f_acm"
          "phy_tegra194_p2u"
          "pcie_tegra194"
          "nvme"
          "nvme-core"
        ];

        # mkBefore: patch flashFromDevice before overlay-with-config bakes it into /init.
        flashFromDeviceNvmeOverlay = final: prev: {
          nvidia-jetpack6 = prev.nvidia-jetpack6.overrideScope (_jfinal: jprev: {
            flashFromDevice = final.callPackage ./pkgs/flash-from-device-nvme.nix {
              flashFromDevice = jprev.flashFromDevice;
            };
          });
        };

        # mkAfter: override nvidia-jetpack (not only nvidia-jetpack6) so initrdFlashScript
        # sees the fixed signedFirmware. makeInitrd packs that closure — do not repack.
        signedFirmwareFixupOverlay = final: prev: {
          nvidia-jetpack = prev.nvidia-jetpack.overrideScope (_jfinal: jprev: {
            signedFirmware =
              final.runCommand "signed-${jprev.l4tMajorMinorPatchVersion}-aleph" {
                nativeBuildInputs = [final.buildPackages.python3];
              } ''
                bash ${./pkgs/fixup-signed-firmware.sh} \
                  ${jprev.signedFirmware} $out ${espImage} ${rootImage}
              '';
          });
        };

        flash-initrd-cross = flashToolSystem.extendModules {
          modules = [
            {nixpkgs.buildPlatform.system = "x86_64-linux";}
            {nixpkgs.overlays = [secureBzip2Overlay];}
            ({lib, ...}: {
              nixpkgs.overlays = lib.mkBefore [flashFromDeviceNvmeOverlay];
            })
            ({lib, ...}: {
              nixpkgs.overlays = lib.mkAfter [signedFirmwareFixupOverlay];
            })
            ({
              lib,
              pkgs,
              ...
            }: {
              hardware.nvidia-jetpack.firmware.initialBootOrder = ["nvme" "usb" "emmc" "sd" "scsi"];

              hardware.nvidia-jetpack.flashScriptOverrides = {
                additionalInitrdFlashModules = lib.mkForce flashInitrdModules;

                partitionTemplate =
                  pkgs.runCommand "aleph-t234-qspi-nvme.xml" {
                    nativeBuildInputs = [pkgs.buildPackages.xmlstarlet];
                    inherit rootImage;
                  } ''
                    app_size=$(stat -c %s "$rootImage")
                    app_size=$(( (app_size + 1048575) / 1048576 * 1048576 ))
                    # instance=0 → tegraflash emits mbr_12_0.bin / gpt_*_12_0.bin
                    # APPFILE + allocation 0x8: keep system.img under NO_ROOTFS=1
                    xmlstarlet ed \
                      -d '//device[@type="nvme"]/partition[not(@name="master_boot_record" or @name="primary_gpt" or @name="esp" or @name="APP" or @name="secondary_gpt")]' \
                      -u '//device[@type="nvme"]/@instance' -v 0 \
                      -i '//device[@type="nvme"]/partition[@name="esp" and not(@id)]' -t attr -n id -v 1 \
                      -u '//device[@type="nvme"]/partition[@name="esp"]/size' -v 536870912 \
                      -u '//device[@type="nvme"]/partition[@name="esp"]/filename' -v 'esp.img' \
                      -u '//device[@type="nvme"]/partition[@name="APP"]/@id' -v 2 \
                      -u '//device[@type="nvme"]/partition[@name="APP"]/size' -v "$app_size" \
                      -u '//device[@type="nvme"]/partition[@name="APP"]/filename' -v 'APPFILE' \
                      -u '//device[@type="nvme"]/partition[@name="APP"]/allocation_attribute' -v '0x8' \
                      -u '//device[@type="nvme"]/partition[@name="APP"]/unique_guid' -v '${rootPartitionUUID}' \
                      ${pkgs.nvidia-jetpack.bspSrc}/bootloader/generic/cfg/flash_t234_qspi_nvme.xml \
                      >$out
                  '';

                # flash.sh --external-device is a boolean (unlike l4t_initrd_flash.sh).
                flashArgs = lib.mkForce [
                  "--external-device"
                  "jetson-orin-nano-devkit"
                  "external"
                ];

                preFlashCommands = ''
                  cp -v ${espImage} bootloader/esp.img
                  cp -v ${rootImage} bootloader/system.img
                '';

                postPatch = ''
                  cp ${./tegra234-mb2-bct-misc-p3767-0000.dts} bootloader/generic/BCT/tegra234-mb2-bct-misc-p3767-0000.dts
                '';
              };
            })
          ];
        };

        flash-initrd-bin-name = "initrd-flash-${flash-initrd-cross.config.hardware.nvidia-jetpack.name}";
        hostPkgs = nixpkgs.legacyPackages.x86_64-linux;
      in {
        flash-uefi = hostPkgs.runCommand "flash-uefi" {} ''
          mkdir -p $out
          cp ${flash-cross.config.system.build.legacyFlashScript}/bin/flash-orin-nx-devkit $out/flash-uefi
          sed -i '46i\cp ${./tegra234-mb2-bct-misc-p3767-0000.dts} bootloader/generic/BCT/tegra234-mb2-bct-misc-p3767-0000.dts' $out/flash-uefi
          chmod +x $out/flash-uefi
        '';

        flash-initrd = hostPkgs.runCommand "flash-initrd" {} ''
          mkdir -p $out
          cp ${flash-initrd-cross.config.system.build.initrdFlashScript}/bin/${flash-initrd-bin-name} $out/flash-initrd
          chmod +x $out/flash-initrd
        '';
      };
      lib.installerSystem = installerSystem;
      templates.default = {
        path = ./template;
        description = "custom nixos image";
      };
    };
}
