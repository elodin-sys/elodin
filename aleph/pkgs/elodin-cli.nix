{
  pkgs,
  lib,
  rustToolchain,
  ...
}: let
  # The Aleph sets nixpkgs.config.cudaSupport = true (aleph-cuda.nix), which
  # flips jax's default to pull the CUDA plugin (-> nccl, unsupported on
  # aarch64). Build jax without CUDA; the sim uses the cranelift backend and the
  # renderer uses Vulkan, so GPU jax is not needed on-device.
  pythonPackages = pkgs.python313Packages.overrideScope (_final: prev: {
    jax = prev.jax.override {cudaSupport = false;};
  });
  elodinPy = pkgs.callPackage ../../nix/pkgs/elodin-py.nix {
    inherit rustToolchain pythonPackages;
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
