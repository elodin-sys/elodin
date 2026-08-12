{
  lib,
  pkgs,
  ...
}: let
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
  mesaVulkanIcdPath =
    lib.concatMapStringsSep ":" (
      manifest: "${pkgs.mesa}/share/vulkan/icd.d/${manifest}_icd.${pkgs.stdenv.hostPlatform.uname.processor}.json"
    ) [
      "radeon"
      "intel"
      "lvp"
    ];
  gpuDetectScript = pkgs.writeShellScript "elodin-gpu-detect" ''
    set -u
    shopt -s nullglob

    mode="''${ELODIN_GPU:-auto}"
    nvidia_icd=""
    for candidate in \
      /run/opengl-driver/share/vulkan/icd.d/nvidia_icd*.json \
      /usr/share/vulkan/icd.d/nvidia_icd*.json \
      /etc/vulkan/icd.d/nvidia_icd*.json; do
      if [ -e "$candidate" ]; then
        nvidia_icd="$candidate"
        break
      fi
    done

    nvidia_present() {
      [ -e /proc/driver/nvidia/version ] && [ -n "$nvidia_icd" ]
    }

    # A Mesa-capable GPU (Intel 0x8086 / AMD 0x1002) means the editor has a
    # working non-NVIDIA path, so do not force the host NVIDIA driver. Match
    # those vendors explicitly: BMC/IPMI framebuffers (e.g. ASPEED 0x1a03,
    # Matrox 0x102b) also expose render nodes but are not performance GPUs, so a
    # "not 0x10de" test would wrongly skip the hook on a Quadro-only server.
    has_mesa_gpu() {
      for vendor in /sys/class/drm/renderD*/device/vendor; do
        case "$(cat "$vendor" 2>/dev/null)" in
          0x8086 | 0x1002) return 0 ;;
        esac
      done
      return 1
    }

    use_nvidia=0
    case "$mode" in
      nvidia)
        if nvidia_present; then
          use_nvidia=1
        fi
        ;;
      mesa)
        use_nvidia=0
        ;;
      *)
        if nvidia_present && ! has_mesa_gpu; then
          use_nvidia=1
        fi
        ;;
    esac

    [ "$use_nvidia" = 1 ] || exit 0

    nvidia_lib_dir="''${TMPDIR:-/tmp}/elodin-nvidia-libs"
    mkdir -p "$nvidia_lib_dir"
    have_glx=0
    declare -A linked_libs=()
    for lib_dir in \
      /run/opengl-driver/lib \
      /usr/lib/x86_64-linux-gnu \
      /usr/lib/aarch64-linux-gnu \
      /usr/lib64; do
      for lib in "$lib_dir"/libGLX_nvidia.so* \
        "$lib_dir"/libEGL_nvidia.so* \
        "$lib_dir"/libcuda.so* \
        "$lib_dir"/libnvcuvid.so* \
        "$lib_dir"/libnvidia-*.so*; do
        [ -e "$lib" ] || continue
        lib_name="$(basename "$lib")"
        # Keep the first match so later fallback directories cannot replace it.
        [ -z "''${linked_libs[$lib_name]+x}" ] || continue
        ln -sf "$lib" "$nvidia_lib_dir/$lib_name"
        linked_libs["$lib_name"]=1
        case "$lib" in
          */libGLX_nvidia.so*) have_glx=1 ;;
        esac
      done
    done

    # libGLX_nvidia provides both the NVIDIA Vulkan ICD (per nvidia_icd.json) and
    # the GLVND GLX vendor library. Without it, exporting the NVIDIA-only ICD and
    # __GLX_VENDOR_LIBRARY_NAME would just break GL/Vulkan, so keep the Mesa
    # defaults when the host driver libraries are not present.
    [ "$have_glx" = 1 ] || exit 0

    # The driver links against the host libX11/libxcb, and having two Xlib
    # copies in one process (Nix's and the host's) crashes inside the driver, so
    # route the whole process at the host X libraries. glibc is deliberately
    # left alone: the host libc is older than Nix's and loading it breaks
    # symbol resolution for everything else.
    for lib_dir in \
      /run/opengl-driver/lib \
      /usr/lib/x86_64-linux-gnu \
      /usr/lib/aarch64-linux-gnu \
      /usr/lib64; do
      for lib in \
        "$lib_dir"/libX11.so.6 \
        "$lib_dir"/libXext.so.6 \
        "$lib_dir"/libXau.so.6 \
        "$lib_dir"/libXdmcp.so.6 \
        "$lib_dir"/libbsd.so.0 \
        "$lib_dir"/libmd.so.0 \
        "$lib_dir"/libxcb.so.1 \
        "$lib_dir"/libxcb-*.so.*; do
        [ -e "$lib" ] || continue
        lib_name="$(basename "$lib")"
        [ -z "''${linked_libs[$lib_name]+x}" ] || continue
        ln -sf "$lib" "$nvidia_lib_dir/$lib_name"
        linked_libs["$lib_name"]=1
      done
    done

    printf 'export VK_ICD_FILENAMES=%s\n' "$nvidia_icd"
    printf 'export VK_DRIVER_FILES=%s\n' "$nvidia_icd"
    printf 'export __GLX_VENDOR_LIBRARY_NAME=nvidia\n'
    printf 'export LD_LIBRARY_PATH="%s''${LD_LIBRARY_PATH:+:''${LD_LIBRARY_PATH}}"\n' "$nvidia_lib_dir"
    printf 'unset LIBGL_ALWAYS_SOFTWARE\n'
  '';
  nvidiaHookScript = pkgs.writeShellScript "elodin-gpu-hook" ''
    eval "$(${gpuDetectScript})"
  '';
in {
  inherit alsaPluginDir asoundConf mesaVulkanIcdPath nvidiaHookScript;

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
  linuxGraphicsEnv = {pkgs}: {
    LIBGL_DRIVERS_PATH = "${pkgs.mesa}/lib/dri";
    __GLX_VENDOR_LIBRARY_NAME = "mesa";
    LIBVA_DRIVERS_PATH = "${pkgs.mesa}/lib/dri";
    VK_ICD_FILENAMES = mesaVulkanIcdPath;
    VK_DRIVER_FILES = mesaVulkanIcdPath;
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
    graphicsWrapperArgs ? null,
  }: let
    linuxLibPath = lib.optionalString pkgs.stdenv.isLinux (
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
      ])
    );
    defaultGraphicsWrapperArgs = lib.optionalString pkgs.stdenv.isLinux ''
      --prefix LD_LIBRARY_PATH : "${linuxLibPath}" \
      --set LIBGL_DRIVERS_PATH "${pkgs.mesa}/lib/dri" \
      --set __GLX_VENDOR_LIBRARY_NAME "mesa" \
      --set LIBVA_DRIVERS_PATH "${pkgs.mesa}/lib/dri" \
      --set VK_ICD_FILENAMES "${mesaVulkanIcdPath}" \
      --set VK_DRIVER_FILES "${mesaVulkanIcdPath}" \
      --prefix VK_LAYER_PATH : "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d" \
      --set ALSA_PLUGIN_DIR "${alsaPluginDir}/lib/alsa-lib" \
      --set ALSA_CONFIG_PATH "${asoundConf}" \
      --run "source ${nvidiaHookScript}"
    '';
    graphicsArgs =
      if graphicsWrapperArgs == null
      then defaultGraphicsWrapperArgs
      else graphicsWrapperArgs;
  in ''
    --prefix PATH : "${python}/bin" \
    --prefix PYTHONPATH : "${pythonPath}" \
    --prefix PYTHONPATH : "${python}/lib/python${pythonMajorMinor}" \
    ${graphicsArgs}
  '';

  # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
  netlibWorkaround = lib.optionalString pkgs.stdenv.isLinux "-Wno-error=incompatible-pointer-types";
}
