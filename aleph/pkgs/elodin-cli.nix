{
  pkgs,
  lib,
  rustToolchain,
  ...
}: let
  # jax is forced CPU by the aleph overlay (pythonPackagesExtensions); cudaSupport
  # (aleph-cuda.nix) would otherwise pull jax-cuda12-plugin -> nccl (unused).
  elodinPy = pkgs.callPackage ../../nix/pkgs/elodin-py.nix {
    inherit rustToolchain;
    pythonPackages = pkgs.python313Packages;
    python = pkgs.python313;
  };
  common = pkgs.callPackage ../../nix/pkgs/common.nix {};
  nvidiaIcd = "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json";
  jetsonLibPath = lib.makeLibraryPath (with pkgs; [
    alsa-lib
    libpulseaudio
    pipewire
    vulkan-loader
    vulkan-validation-layers
    libglvnd
    libdrm
    libxkbcommon
    udev
    systemd
  ]);
in
  pkgs.callPackage ../../nix/pkgs/elodin-cli.nix {
    inherit rustToolchain;
    elodinPy = elodinPy.py;
    python = elodinPy.python;
    pythonPackages = elodinPy.pythonPackages;
    graphicsWrapperArgs = ''
      --prefix LD_LIBRARY_PATH : "/run/opengl-driver/lib:${jetsonLibPath}" \
      --set-default LIBGL_DRIVERS_PATH "/run/opengl-driver/lib/dri:${pkgs.mesa}/lib/dri" \
      --set-default LIBVA_DRIVERS_PATH "/run/opengl-driver/lib/dri:${pkgs.mesa}/lib/dri" \
      --set VK_ICD_FILENAMES "${nvidiaIcd}" \
      --set VK_DRIVER_FILES "${nvidiaIcd}" \
      --prefix VK_LAYER_PATH : "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d" \
      --set-default ELODIN_ASSETS_DIR "/var/lib/elodin/assets" \
      --set ALSA_PLUGIN_DIR "${common.alsaPluginDir}/lib/alsa-lib" \
      --set ALSA_CONFIG_PATH "${common.asoundConf}"
    '';
  }
