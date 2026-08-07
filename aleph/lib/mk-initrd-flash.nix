{
  nixpkgs,
  alephSystem,
  baseModules,
  secureBzip2Overlay,
}: let
  hostPkgs = nixpkgs.legacyPackages.x86_64-linux;

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

  # MB1 rejects RCM blobs between 383 MB and 803 MB (err 0x354b0107),
  # so OS images are sideloaded over a USB ethernet gadget.
  sideloadUrl = "http://192.168.7.1:8080";
  flashPayload = hostPkgs.runCommand "aleph-flash-payload" {} ''
    mkdir -p $out
    gzip -1 -c ${espImage} > $out/esp.img.gz
    gzip -1 -c ${rootImage} > $out/system.img.gz
  '';

  # The flash initrd /init resolves flashFromDevice and signedFirmware through
  # the nvidia-jetpack scope fixpoint, so one late override of that scope
  # regenerates the initrd with both Aleph pieces (mkAfter: must compose after
  # jetpack's overlay-with-config).
  alephFlashOverlay = final: prev: {
    nvidia-jetpack = prev.nvidia-jetpack.overrideScope (_jfinal: jprev: {
      # jetpack's on-device flasher skips NVMe rows; patch in 12:0 writes
      flashFromDevice =
        final.runCommand "flash-from-device" {
          meta.mainProgram = "flash-from-device";
          nativeBuildInputs = [final.buildPackages.python3];
        } ''
          mkdir -p $out/bin
          cp ${final.lib.getExe jprev.flashFromDevice} $out/bin/flash-from-device
          chmod +w $out/bin/flash-from-device
          python3 ${../pkgs/patch-flash-from-device-nvme.py} $out/bin/flash-from-device
          chmod +x $out/bin/flash-from-device
        '';
      # Strip OS images from the firmware tree (kept out of the RCM blob) and
      # point flash.idx esp/APP rows at the host sideload URLs.
      # makeInitrd packs this closure — do not repack.
      signedFirmware =
        final.runCommand "signed-${jprev.l4tMajorMinorPatchVersion}-aleph" {
          nativeBuildInputs = [final.buildPackages.python3];
        } ''
          bash ${../pkgs/fixup-signed-firmware.sh} \
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
        # g_ncm is builtin and would claim the UDC before the flash gadget.
        hardware.nvidia-jetpack.console.args = lib.mkAfter ["initcall_blacklist=ncm_driver_init"];

        hardware.nvidia-jetpack.flashScriptOverrides = {
          # jetpack adds its own QSPI/USB-gadget modules; these are the Aleph
          # additions for the SSD. mkForce sheds the option default
          # (availableKernelModules), which lists modules absent from
          # tegra_defconfig and fails makeModulesClosure (allowMissing = false).
          # NVMe and ECM are builtin; only the PCIe drivers need packaging.
          additionalInitrdFlashModules = lib.mkForce [
            "phy_tegra194_p2u"
            "pcie_tegra194"
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
            cp ${../tegra234-mb2-bct-misc-p3767-0000.dts} bootloader/generic/BCT/tegra234-mb2-bct-misc-p3767-0000.dts
            ${pkgs.buildPackages.python3}/bin/python3 ${../pkgs/patch-tegraflash-storage-info.py} bootloader/tegraflash_impl_t234.py
          '';
        };
      })
    ];
  };

  hostHooks = hostPkgs.replaceVars ../pkgs/flash-initrd-host-hooks.sh {
    ip = "${hostPkgs.iproute2}/bin/ip";
    python = "${hostPkgs.python3}/bin/python3";
    inherit flashPayload;
  };
  # Keep headroom below MB1's measured RCM blob limit.
  maxInitrdBytes = 314572800; # 300 MiB
in
  hostPkgs.runCommand "flash-initrd" {} ''
    initrd=${flash-initrd-cross.pkgs.nvidia-jetpack.flashInitrd}/initrd
    size=$(stat -c %s "$initrd")
    if [ "$size" -ge ${toString maxInitrdBytes} ]; then
      echo "error: flash initrd is $size bytes (≥ ${toString maxInitrdBytes});" >&2
      echo "MB1 rejects RCM blobs above ~0.4 GB (RCM_BLOB err 0x354b0107)." >&2
      exit 1
    fi
    mkdir -p $out
    cp ${flash-initrd-cross.config.system.build.initrdFlashScript}/bin/initrd-flash-${flash-initrd-cross.config.hardware.nvidia-jetpack.name} $out/flash-initrd
    chmod +w $out/flash-initrd

    printf '%s\n' 'source ${hostHooks}' aleph_flash_prepare_host > insert-pre
    echo aleph_flash_start_sideload > insert-post

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
    chmod +x $out/flash-initrd
  ''
