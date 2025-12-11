{ applyPatches
, bspSrc
, buildPackages
, gitRepos
, kernel
, l4tMajorMinorPatchVersion
, lib
, runCommand
, stdenv
, ...
}:
let
  patchedBsp = applyPatches {
    name = "patchedBsp";
    src = bspSrc;
    patches = [
      ./Makefile.diff
    ];
  };

  mkCopyProjectCommand = project: ''
    mkdir -p "$out/${project.name}"
    cp --no-preserve=all -vr "${project}"/. "$out/${project.name}"
  '';

  l4t-oot-projects = [
    (gitRepos.hwpm.overrideAttrs { name = "hwpm"; })
    (applyPatches {
      name = "nvidia-oot";
      src = gitRepos.nvidia-oot;
      patches = [
        ./0001-rtl8822ce-Fix-Werror-address.patch
        ./0002-sound-Fix-include-path-for-tegra-virt-alt-include.patch
      ];
    })
    (gitRepos.nvgpu.overrideAttrs { name = "nvgpu"; })
    (applyPatches {
      name = "nvdisplay";
      src = gitRepos.nvdisplay;
      patches = [
        ./0001-nvidia-drm-Guard-nv_dev-in-nv_drm_suspend_resume.patch
        ./0002-ANDURIL-Add-some-missing-BASE_CFLAGS.patch
        ./0003-ANDURIL-Update-drm_gem_object_vmap_has_map_arg-test.patch
        ./0004-ANDURIL-override-KERNEL_SOURCES-and-KERNEL_OUTPUT-if.patch
      ];
    })
    (applyPatches {
      name = "nvethernetrm";
      src = gitRepos.nvethernetrm;
      # Some directories in the git repo are RO.
      # This works for L4T b/c they use different output directory
      postPatch = ''
        chmod -R u+w osi
      '';
    })
  ];

  l4t-oot-modules-sources = runCommand "l4t-oot-sources" { }
    (
      # Copy the Makefile
      ''
        mkdir -p "$out"
        cp "${patchedBsp}/source/Makefile" "$out/Makefile"
      ''
      # copy the projects
      + lib.strings.concatMapStringsSep "\n" mkCopyProjectCommand l4t-oot-projects
      # See bspSrc/source/source_sync.sh symlink at end of file
      + ''
        ln -vsrf "$out/nvethernetrm" "$out/nvidia-oot/drivers/net/ethernet/nvidia/nvethernet/nvethernetrm"
      ''
    );
in
stdenv.mkDerivation {
  __structuredAttrs = true;
  strictDeps = true;

  pname = "l4t-oot-modules";
  version = "${l4tMajorMinorPatchVersion}";
  src = l4t-oot-modules-sources;

  nativeBuildInputs = kernel.moduleBuildDependencies;
  depsBuildBuild = [ buildPackages.stdenv.cc ];

  # See bspSrc/source/Makefile
  # We can't use kernelModuleMakeFlags because it sets KBUILD_OUTPUT, which nvdisplay won't like. DON'T DO IT!
  makeFlags = kernel.commonMakeFlags ++ [
    "KERNEL_HEADERS=${kernel.dev}/lib/modules/${kernel.modDirVersion}/source"
    "KERNEL_OUTPUT=${kernel.dev}/lib/modules/${kernel.modDirVersion}/build"
    "INSTALL_MOD_PATH=$(out)"
    "IGNORE_PREEMPT_RT_PRESENCE=1"
  ];

  postInstall = ''
    mkdir -p $dev
    cat **/Module.symvers > $dev/Module.symvers

    mkdir -p $dev/include/nvidia
    install -m 0644 out/nvidia-conftest/nvidia/conftest.h $dev/include/nvidia/
  '';

  outputs = [
    "out"
    "dev"
  ];

  # # GCC 14.2 seems confused about DRM_MODESET_LOCK_ALL_BEGIN/DRM_MODESET_LOCK_ALL_END in nvdisplay/kernel-open/nvidia-drm/nvidia-drm-drv.c:1344
  # extraMakeFlags = [ "KCFLAGS=-Wno-error=unused-label" ];

  buildFlags = [ "modules" ];
  installTargets = [ "modules_install" ];
}
