{ applyPatches
, lib
, fetchFromGitHub
, l4t-xusb-firmware
, realtime ? false
, kernelPatches ? [ ]
, structuredExtraConfig ? { }
, argsOverride ? { }
, buildLinux
, gitRepos
, ...
}@args:
buildLinux (args // {
  # See Makefile in kernel source root for VERSION/PATCHLEVEL/SUBLEVEL.
  version = "5.15.148";
  extraMeta.branch = "5.15";

  defconfig = "defconfig";
  #defconfig = "defconfig_aleph_prev";
  #defconfig = "p3767_antmicro_job_defconfig";

  # https://github.com/NixOS/nixpkgs/pull/366004
  # introduced a breaking change that if a module is declared but it is not being used it will fail
  # if you try to suppress each of he errors e.g.
  # REISERFS_FS_SECURITY = lib.mkForce unset; within structuredExtraConfig
  # that list runs to a long 100+ modules so we go back to the previous default and ignore them
  ignoreConfigErrors = true;

  # disabling the dependency on the common-config would seem appropriate as we define our own defconfig
  # however, it seems that some of the settings for e.g. fw loading are only made available there.
  # TODO: a future task could be to set this, disable ignoreConfigErrors and add the needed modules to the
  # structuredExtraConfig below.
  #enableCommonConfig = false;

  # Using applyPatches here since it's not obvious how to append an extra
  # postPatch. This is not very efficient.
  src = gitRepos."kernel/kernel-jammy-src";
  autoModules = false;
  features = { }; # TODO: Why is this needed in nixpkgs master (but not NixOS 22.05)?

  kernelPatches = [
    {
      name = "ipu: Depend on x86";
      patch = ./0001-ipu-Depend-on-x86.patch;
    }
  ] ++ kernelPatches;

  structuredExtraConfig = with lib.kernel; {
    #  MODPOST modules-only.symvers
    #ERROR: modpost: "xhci_hc_died" [drivers/usb/host/xhci-tegra.ko] undefined!
    #ERROR: modpost: "xhci_hub_status_data" [drivers/usb/host/xhci-tegra.ko] undefined!
    #ERROR: modpost: "xhci_enable_usb3_lpm_timeout" [drivers/usb/host/xhci-tegra.ko] undefined!
    #ERROR: modpost: "xhci_hub_control" [drivers/usb/host/xhci-tegra.ko] undefined!
    #ERROR: modpost: "xhci_get_rhub" [drivers/usb/host/xhci-tegra.ko] undefined!
    #ERROR: modpost: "xhci_urb_enqueue" [drivers/usb/host/xhci-tegra.ko] undefined!
    #ERROR: modpost: "xhci_irq" [drivers/usb/host/xhci-tegra.ko] undefined!
    #USB_XHCI_TEGRA = module;
    USB_XHCI_TEGRA = yes;

    # stage-1 links /lib/firmware to the /nix/store path in the initramfs.
    # However, since it's builtin and not a module, that's too late, since
    # the kernel will have already tried loading!
    EXTRA_FIRMWARE_DIR = freeform "${l4t-xusb-firmware}/lib/firmware";
    EXTRA_FIRMWARE = freeform "nvidia/tegra194/xusb.bin";

    # Override the default CMA_SIZE_MBYTES=32M setting in common-config.nix with the default from tegra_defconfig
    # Otherwise, nvidia's driver craps out
    CMA_SIZE_MBYTES = lib.mkForce (freeform "64");

    ### So nat.service and firewall work ###
    NF_TABLES = module; # This one should probably be in common-config.nix
    # this NFT_NAT is not actually being set. when build with enableCommonConfig = false;
    # and not ignoreConfigErrors = true; it will fail with error about unused option
    # unused means that it wanted to set it as a module, but make oldconfig didn't ask it about that option,
    # so it didn't get a chance to set it.
    NFT_NAT = module;
    NFT_MASQ = module;
    NFT_REJECT = module;
    NFT_COMPAT = module;
    NFT_LOG = module;
    NFT_COUNTER = module;

    # search for "ip46tables" in nixpkgs and find all the -m options.
    # Enable the corresponding Kconfigs
    # TODO: nixpkgs should turn these on themselves.
    NETFILTER_XT_MATCH_PKTTYPE = module;
    NETFILTER_XT_MATCH_COMMENT = module;
    NETFILTER_XT_MATCH_CONNTRACK = module;
    NETFILTER_XT_MATCH_LIMIT = module;
    NETFILTER_XT_MATCH_MARK = module;
    NETFILTER_XT_MATCH_MULTIPORT = module;

    IP_NF_MATCH_RPFILTER = module;

    # IPv6 is enabled by default and without some of these `firewall.service` will explode.
    IP6_NF_MATCH_AH = module;
    IP6_NF_MATCH_EUI64 = module;
    IP6_NF_MATCH_FRAG = module;
    IP6_NF_MATCH_OPTS = module;
    IP6_NF_MATCH_HL = module;
    IP6_NF_MATCH_IPV6HEADER = module;
    IP6_NF_MATCH_MH = module;
    IP6_NF_MATCH_RPFILTER = module;
    IP6_NF_MATCH_RT = module;
    IP6_NF_MATCH_SRH = module;

    # Needed since mdadm stuff is currently unconditionally included in the initrd
    # This will hopefully get changed, see: https://github.com/NixOS/nixpkgs/pull/183314
    MD_LINEAR = module;
    MD_RAID0 = module;
    MD_RAID1 = module;
    MD_RAID10 = module;
    MD_RAID456 = module;

    # Needed for booting from USB
    USB_UAS = module;

    FW_LOADER_COMPRESS_XZ = yes;
    FW_LOADER_COMPRESS_ZSTD = yes;

    # Restore default LSM from security/Kconfig. Undoes Nvidia downstream changes.
    LSM = freeform "landlock,lockdown,yama,loadpin,safesetid,integrity,selinux,smack,tomoyo,apparmor,bpf";
  } // lib.optionalAttrs realtime {
    PREEMPT_VOLUNTARY = lib.mkForce no; # Disable the one set in common-config.nix
    # These are the options enabled/disabled by source/generic_rt_build.sh (this file comes after source/source_sync.sh)
    PREEMPT_RT = yes;
    DEBUG_PREEMPT = no;
    KVM = no;
    EMBEDDED = yes;
    NAMESPACES = yes;
    CPU_IDLE_TEGRA18X = no;
    CPU_FREQ_GOV_INTERACTIVE = no;
    CPU_FREQ_TIMES = no;
    FAIR_GROUP_SCHED = no;
  } // structuredExtraConfig;

} // argsOverride)
