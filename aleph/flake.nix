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
        flashToolSystem = nixpkgs.lib.nixosSystem {
          system = alephSystem;
          modules = [({...}: {imports = builtins.attrValues baseModules;})];
        };
        flashInstallSystem = flashToolSystem.extendModules {
          modules = [{aleph.nvmeImage.enable = true;}];
        };
        espImage = flashInstallSystem.config.system.build.alephEspImage;
        rootImage = flashInstallSystem.config.system.build.alephRootImage;
        rootPartitionUUID = flashInstallSystem.config.aleph.fs.rootPartitionUUID;

        # The device rejects RCM blobs somewhere between 383 MB and 803 MB
        # (MB1 RCM_BLOB err 0x354b0107; see ai-context/rcm-bisect). OS images are
        # therefore sideloaded to the flash initrd over a USB ethernet gadget.
        sideloadUrl = "http://192.168.7.1:8080";
        flashPayload = hostPkgs.runCommand "aleph-flash-payload" {} ''
          mkdir -p $out
          gzip -1 -c ${espImage} > $out/esp.img.gz
          gzip -1 -c ${rootImage} > $out/system.img.gz
        '';

        flash-cross = jetpack.nixosConfigurations."orin-nx-devkit".extendModules {
          modules = [
            {nixpkgs.buildPlatform.system = "x86_64-linux";}
            {nixpkgs.overlays = [secureBzip2Overlay];}
          ];
        };

        # The flash initrd /init resolves flashFromDevice and signedFirmware through
        # the nvidia-jetpack scope fixpoint, so one late override of that scope
        # regenerates the initrd with both Aleph pieces (mkAfter: must compose after
        # jetpack's overlay-with-config).
        alephFlashOverlay = final: prev: {
          nvidia-jetpack = prev.nvidia-jetpack.overrideScope (_jfinal: jprev: {
            # jetpack's on-device flasher skips NVMe rows; patch in 9:0 / 12:* writes
            flashFromDevice =
              final.runCommand "flash-from-device" {
                meta.mainProgram = "flash-from-device";
                nativeBuildInputs = [final.buildPackages.python3];
              } ''
                mkdir -p $out/bin
                cp ${final.lib.getExe jprev.flashFromDevice} $out/bin/flash-from-device
                chmod +w $out/bin/flash-from-device
                python3 ${./pkgs/patch-flash-from-device-nvme.py} $out/bin/flash-from-device
                chmod +x $out/bin/flash-from-device
              '';
            # MB2 initializes every device in its cold-boot BCT storage info; on the
            # Aleph carrier PCIe cannot come up that early (SError in
            # tegrabl_pcie_soc_init, board falls back to recovery). Strip NVMe from
            # the storage-info step only — GPT/flash.idx keep the full layout.
            flash-tools = jprev.flash-tools.overrideAttrs (old: {
              postInstall =
                (old.postInstall or "")
                + ''
                  python3 ${./pkgs/patch-tegraflash-storage-info.py} $out/bootloader/tegraflash_impl_t234.py
                '';
            });
            # Strip OS images from the firmware tree (kept out of the RCM blob) and
            # point flash.idx esp/APP rows at the host sideload URLs.
            # makeInitrd packs this closure — do not repack.
            signedFirmware =
              final.runCommand "signed-${jprev.l4tMajorMinorPatchVersion}-aleph" {
                nativeBuildInputs = [final.buildPackages.python3];
              } ''
                bash ${./pkgs/fixup-signed-firmware.sh} \
                  ${jprev.signedFirmware} $out ${sideloadUrl}
              '';
            # Add an ECM ethernet function + static IP to the flash initrd /init so
            # flash-from-device can fetch the images. The kernel lets later cpio
            # members win, so append a tiny archive instead of repacking.
            flashInitrd =
              final.runCommand "flash-initrd-ecm" {
                nativeBuildInputs = [final.buildPackages.cpio final.buildPackages.gzip];
                passthru = jprev.flashInitrd.passthru;
              } ''
                mkdir work && cd work
                zcat ${jprev.flashInitrd}/initrd | cpio -id init 2>/dev/null
                target=$(readlink init)
                member=''${target#/}
                zcat ${jprev.flashInitrd}/initrd | cpio -id "$member" 2>/dev/null
                rm init
                cp "$member" init
                chmod +w init

                grep -q 'ln -s $gadget/functions/acm.usb0 $gadget/configs/c.1/' init
                sed -i 's|ln -s $gadget/functions/acm.usb0 $gadget/configs/c.1/|&\nmkdir $gadget/functions/ecm.usb0\necho 32:70:05:18:01:01 >$gadget/functions/ecm.usb0/host_addr\necho 32:70:05:18:01:02 >$gadget/functions/ecm.usb0/dev_addr\nln -s $gadget/functions/ecm.usb0 $gadget/configs/c.1/|' init
                grep -q 'mdev -s' init
                sed -i 's|mdev -s|&\nifconfig usb0 192.168.7.2 netmask 255.255.255.0 up|' init

                mkdir -p $out
                echo init | cpio -H newc -o | gzip -1 > extra.cpio.gz
                cat ${jprev.flashInitrd}/initrd extra.cpio.gz > $out/initrd
              '';
          });
        };

        flash-initrd-cross = flashToolSystem.extendModules {
          modules = [
            {nixpkgs.buildPlatform.system = "x86_64-linux";}
            {nixpkgs.overlays = [secureBzip2Overlay];}
            ({
              lib,
              pkgs,
              ...
            }: {
              nixpkgs.overlays = lib.mkAfter [alephFlashOverlay];

              hardware.nvidia-jetpack.firmware.initialBootOrder = ["nvme" "usb" "emmc" "sd" "scsi"];

              hardware.nvidia-jetpack.flashScriptOverrides = {
                # jetpack adds its own QSPI/USB-gadget modules; these are the Aleph
                # additions for the SSD. mkForce sheds the option default
                # (availableKernelModules), which lists modules absent from
                # tegra_defconfig and fails makeModulesClosure (allowMissing = false).
                additionalInitrdFlashModules = lib.mkForce [
                  "phy_tegra194_p2u"
                  "pcie_tegra194"
                  "nvme"
                  "nvme-core"
                  "usb_f_ecm"
                ];

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

        flash-initrd = let
          flashInitrd = flash-initrd-cross.pkgs.nvidia-jetpack.flashInitrd;
          # MB1 rejects RCM blobs between 383 MB (accepted) and 803 MB (rejected):
          # RCM_BLOB err 0x354b0107, see ai-context/rcm-bisect. Keep headroom.
          maxInitrdBytes = 314572800; # 300 MiB
        in
          hostPkgs.runCommand "flash-initrd" {} ''
            initrd=${flashInitrd}/initrd
            size=$(stat -c %s "$initrd")
            if [ "$size" -ge ${toString maxInitrdBytes} ]; then
              echo "error: flash initrd is $size bytes (≥ ${toString maxInitrdBytes});" >&2
              echo "MB1 rejects RCM blobs above ~0.4 GB (RCM_BLOB err 0x354b0107)." >&2
              exit 1
            fi
            mkdir -p $out
            cp ${flash-initrd-cross.config.system.build.initrdFlashScript}/bin/${flash-initrd-bin-name} $out/flash-initrd
            chmod +w $out/flash-initrd

            # Right after cd "$WORKDIR": self-log every run, and raise usbfs staging
            # memory — the 16 MiB default stalls RCM bulk writes with
            # "might be timeout in USB write" (forums.developer.nvidia.com/t/360581).
            cat > insert-pre <<'EOF'
            exec > >(tee /tmp/flash-initrd-$(date +%s).log) 2>&1
            echo 2048 > /sys/module/usbcore/parameters/usbfs_memory_mb || true
            echo -1 > /sys/module/usbcore/parameters/autosuspend || true
            # ModemManager probes the flash ACM console with AT commands; keep it away
            MM_WAS_ACTIVE=0
            if systemctl is-active --quiet ModemManager 2>/dev/null; then
              MM_WAS_ACTIVE=1
              systemctl stop ModemManager || true
            fi
            EOF

            # Right after flash.sh (RCM boot started): bring up the gadget ethernet
            # and serve the OS images for flash-from-device to fetch.
            cat > insert-post <<'EOF'
            echo "Starting sideload server for the flash initrd..."
            for i in $(seq 1 90); do
              ${hostPkgs.iproute2}/bin/ip link show enx327005180101 >/dev/null 2>&1 && break
              sleep 1
            done
            if ! ${hostPkgs.iproute2}/bin/ip link show enx327005180101 >/dev/null 2>&1; then
              echo "ERR: gadget ethernet enx327005180101 did not appear" >&2
              exit 3
            fi
            # NetworkManager tears our static address down mid-flash; unmanage the
            # iface if NM is present, and keep re-asserting the address regardless.
            command -v nmcli >/dev/null 2>&1 && nmcli dev set enx327005180101 managed no || true
            ${hostPkgs.iproute2}/bin/ip addr replace 192.168.7.1/24 dev enx327005180101
            ${hostPkgs.iproute2}/bin/ip link set enx327005180101 up
            (
              while true; do
                ${hostPkgs.iproute2}/bin/ip addr replace 192.168.7.1/24 dev enx327005180101 2>/dev/null
                ${hostPkgs.iproute2}/bin/ip link set enx327005180101 up 2>/dev/null
                sleep 5
              done
            ) &
            KEEPER_PID=$!
            ${hostPkgs.python3}/bin/python3 -m http.server 8080 --bind 192.168.7.1 --directory ${flashPayload} &
            HTTP_PID=$!
            trap 'kill $HTTP_PID $KEEPER_PID 2>/dev/null || true; [ "$MM_WAS_ACTIVE" = 1 ] && systemctl start ModemManager 2>/dev/null; type on_exit >/dev/null 2>&1 && on_exit || true' EXIT
            EOF

            # After the devicetree copy: boot the RCM kernel with Aleph's tree — the
            # devkit DTB leaves the M.2 PCIe link down, so /dev/nvme0n1 never appears.
            cat > insert-dtb <<'EOF'
            cp ${flash-initrd-cross.config.hardware.deviceTree.package}/${flash-initrd-cross.config.hardware.deviceTree.name} \
              "kernel/dtb/tegra234-p3768-0000+p3767-0000-nv.dtb"
            EOF

            grep -q '^cd "$WORKDIR"$' $out/flash-initrd
            sed -i '/^cd "$WORKDIR"$/r insert-pre' $out/flash-initrd
            grep -q '\. kernel/dtb/$' $out/flash-initrd
            sed -i '/\. kernel\/dtb\/$/r insert-dtb' $out/flash-initrd
            grep -q '^\./flash\.sh ' $out/flash-initrd
            sed -i '/^\.\/flash\.sh /r insert-post' $out/flash-initrd
            # The Aleph kernel builds g_ncm builtin (CONFIG_USB_G_NCM=y, used by
            # usb-eth.nix); it binds the sole UDC before /init runs, so the flash
            # gadget (ACM serial + ECM) can never attach. Blacklist its initcall
            # for the RCM boot only — the installed system is unaffected.
            grep -q '^export CMDLINE=' $out/flash-initrd
            sed -i 's|^export CMDLINE="\(.*\)"|export CMDLINE="\1 initcall_blacklist=ncm_driver_init"|' $out/flash-initrd
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
