{
  lib,
  pkgs,
  ...
}: {
  src = let
    includeSrc = orig_path: type: let
      path = toString orig_path;
      base = baseNameOf path;
      relPath = lib.removePrefix (toString ../..) orig_path;
      matchesPrefix = lib.any (prefix: lib.hasPrefix prefix relPath) [
        "/apps"
        "/libs"
        "/fsw"
        "/examples"
        "/.config"
      ];
      matchesSuffix = lib.any (suffix: lib.hasSuffix suffix base) [
        "Cargo.toml"
        "Cargo.lock"
        "rust-toolchain.toml"
        "rustfmt.toml"
        "logo.txt"
        "logo.png"
        ".rs"
        ".c"
        ".h"
        ".cpp"
        ".hpp"
        ".jinja"
      ];
    in
      (type == "directory" && matchesPrefix) || matchesSuffix;
  in
    lib.cleanSourceWith {
      src = ../..;
      filter = path: type: includeSrc path type;
    };

  # Common Linux graphics and audio dependencies
  linuxGraphicsAudioDeps = with pkgs; [
    # Audio
    alsa-lib
    alsa-lib.dev
    alsa-plugins
    libpulseaudio
    pulseaudio
    pipewire

    # Graphics - Core
    libGL
    libglvnd
    mesa
    libdrm

    # Vulkan
    vulkan-loader
    vulkan-headers
    vulkan-validation-layers
    vulkan-tools

    # X11
    libx11
    libxcursor
    libxrandr
    libxi
    libxext
    libxshmfence

    # Wayland
    wayland
    libxkbcommon
    libxkbcommon.dev

    # Other
    udev
    systemd # For libudev
  ];

  # Common macOS dependencies
  darwinDeps = with pkgs; [
    libiconv
  ];

  # Common build dependencies
  commonBuildInputs = with pkgs; [
    openssl
    openblas
    xz
    zstd
    python313
    gfortran.cc.lib
    # Expose GNU tar as `gnutar` so tests/scripts can call it without shadowing bsdtar `tar`.
    (writeShellScriptBin "gtar" ''exec ${gnutar}/bin/tar "$@"'')
  ];

  # Common native build inputs
  commonNativeBuildInputs = with pkgs; [
    pkg-config
    cmake
    gfortran
    gcc
  ];

  # nixpkgs currently marks ktx-tools as Linux-only, but KTX-Software builds and
  # runs on Darwin. The editor needs toktx at runtime for generated KTX2 skyboxes.
  ktxTools = pkgs.ktx-tools.overrideAttrs (old: {
    meta =
      old.meta
      // {
        platforms = old.meta.platforms ++ lib.platforms.darwin;
      };
  });

  # Function to create Linux library path
  makeLinuxLibraryPath = {pkgs}:
    lib.makeLibraryPath (with pkgs; [
      # Audio
      alsa-lib
      libpulseaudio
      pipewire

      # Graphics - Core
      libGL
      libglvnd
      mesa
      libdrm

      # Vulkan
      vulkan-loader
      vulkan-validation-layers

      # X11
      libx11
      libxcursor
      libxrandr
      libxi
      libxext
      libxshmfence

      # Wayland
      wayland
      libxkbcommon

      # Other
      udev
      systemd
      gfortran.cc.lib
    ]);

  # Linux graphics environment variables
  linuxGraphicsEnv = {pkgs}: let
    alsaPluginDir = pkgs.symlinkJoin {
      name = "elodin-alsa-plugins";
      paths = [
        pkgs.alsa-plugins
        pkgs.pipewire
      ];
    };
    asoundConf = pkgs.writeText "elodin-asound.conf" ''
      <${pkgs.alsa-lib}/share/alsa/alsa.conf>

      pcm.!default {
        type pulse
      }

      ctl.!default {
        type pulse
      }
    '';
  in {
    LIBGL_DRIVERS_PATH = "${pkgs.mesa}/lib/dri";
    __GLX_VENDOR_LIBRARY_NAME = "mesa";
    LIBVA_DRIVERS_PATH = "${pkgs.mesa}/lib/dri";
    VK_ICD_FILENAMES = "${pkgs.mesa}/share/vulkan/icd.d/radeon_icd.x86_64.json:${pkgs.mesa}/share/vulkan/icd.d/intel_icd.x86_64.json:${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json";
    VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
    ALSA_PLUGIN_DIR = "${alsaPluginDir}/lib/alsa-lib";
    ALSA_CONFIG_PATH = "${asoundConf}";
  };

  # Common wrapper arguments for executables
  makeWrapperArgs = {
    pkgs,
    python,
    pythonPath,
    pythonMajorMinor,
  }: let
    linuxLibPath = lib.optionalString pkgs.stdenv.isLinux (
      lib.makeLibraryPath (with pkgs; [
        # Audio
        alsa-lib
        pipewire

        # Graphics - Core
        libGL
        libglvnd
        mesa
        libdrm

        # Vulkan
        vulkan-loader
        vulkan-validation-layers

        # X11
        libx11
        libxcursor
        libxrandr
        libxi
        libxext
        libxshmfence

        # Wayland
        wayland
        libxkbcommon

        # Other
        udev
        systemd
      ])
    );
    graphicsEnv = lib.optionalString pkgs.stdenv.isLinux ''
      --prefix LD_LIBRARY_PATH : "${linuxLibPath}" \
      --set LIBGL_DRIVERS_PATH "${pkgs.mesa}/lib/dri" \
      --set __GLX_VENDOR_LIBRARY_NAME "mesa" \
      --set LIBVA_DRIVERS_PATH "${pkgs.mesa}/lib/dri" \
      --prefix VK_ICD_FILENAMES : "${pkgs.mesa}/share/vulkan/icd.d/radeon_icd.x86_64.json:${pkgs.mesa}/share/vulkan/icd.d/intel_icd.x86_64.json:${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json" \
      --prefix VK_LAYER_PATH : "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d" \
      --set ALSA_PLUGIN_DIR "${pkgs.pipewire}/lib/alsa-lib"
    '';
  in ''
    --prefix PATH : "${python}/bin" \
    --prefix PYTHONPATH : "${pythonPath}" \
    --prefix PYTHONPATH : "${python}/lib/python${pythonMajorMinor}" \
    ${graphicsEnv}
  '';

  # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
  netlibWorkaround = lib.optionalString pkgs.stdenv.isLinux "-Wno-error=incompatible-pointer-types";
}
