{
  pkgs,
  crane,
  rustToolchain,
  lib,
  elodinPy,
  python,
  pythonPackages,
  ...
}: let
  # Direct Rust build using rustPlatform.buildRustPackage
  #
  # We bypass crane entirely due to path resolution issues with the pinned
  # crane version (dfd9a8dfd...) on macOS that cause "crane-utils" build failures.
  # For consistency and simplicity, we use the same direct build approach for
  # both macOS and Linux platforms.
  pname = "elodin";
  # Derive version from Cargo.toml workspace configuration
  cargoToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = cargoToml.workspace.package.version;
  src = pkgs.nix-gitignore.gitignoreSource [] ../..;
  pythonPath = pythonPackages.makePythonPath [elodinPy];
  pythonMajorMinor = lib.versions.majorMinor python.version;

  bin = pkgs.rustPlatform.buildRustPackage rec {
    inherit pname version src;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "apps/elodin";

    nativeBuildInputs = with pkgs; [
      (rustToolchain pkgs)
      pkg-config
      cmake
      makeWrapper # Required for wrapProgram in postInstall
      gfortran # Fortran compiler needed at build time for netlib-src
      gcc # g++ equivalent in nix
    ];

    buildInputs = with pkgs;
      [
        python
        openssl # libssl-dev in ubuntu
        openblas # libopenblas-dev in ubuntu
        xz # liblzma-dev in ubuntu
      ]
      ++ lib.optionals pkgs.stdenv.isDarwin [
        libiconv
        darwin.apple_sdk.frameworks.Security
        darwin.apple_sdk.frameworks.CoreServices
        darwin.apple_sdk.frameworks.SystemConfiguration
      ]
      ++ lib.optionals pkgs.stdenv.isLinux [
        # Audio
        alsa-lib
        alsa-lib.dev
        pipewire

        # Graphics - Core
        libGL
        libglvnd
        mesa

        # Vulkan
        vulkan-loader
        vulkan-headers
        vulkan-validation-layers
        vulkan-tools

        # X11
        xorg.libX11
        xorg.libXcursor
        xorg.libXrandr
        xorg.libXi
        xorg.libXext

        # Wayland
        wayland
        libxkbcommon
        libxkbcommon.dev

        # Other
        udev
        systemd # For libudev
      ];

    doCheck = false;

    postInstall = let
      linuxLibPath = lib.optionalString pkgs.stdenv.isLinux (
        lib.makeLibraryPath [
          # Audio
          pkgs.alsa-lib
          pkgs.pipewire

          # Graphics - Core
          pkgs.libGL
          pkgs.libglvnd # Provides libEGL
          pkgs.mesa # Provides DRI drivers
          pkgs.libdrm

          # Vulkan
          pkgs.vulkan-loader
          pkgs.vulkan-validation-layers

          # X11
          pkgs.xorg.libX11
          pkgs.xorg.libXcursor
          pkgs.xorg.libXrandr
          pkgs.xorg.libXi
          pkgs.xorg.libXext
          pkgs.xorg.libxshmfence

          # Wayland
          pkgs.wayland
          pkgs.libxkbcommon

          # Other
          pkgs.udev
          pkgs.systemd # For libudev
        ]
      );
    in ''
      wrapProgram $out/bin/elodin \
        --prefix PATH : "${python}/bin" \
        --prefix PYTHONPATH : "${pythonPath}" \
        --prefix PYTHONPATH : "${python}/lib/python${pythonMajorMinor}" \
        ${lib.optionalString pkgs.stdenv.isLinux ''
        --prefix LD_LIBRARY_PATH : "${linuxLibPath}" \
        --set LIBGL_DRIVERS_PATH "${pkgs.mesa}/lib/dri" \
        --set __GLX_VENDOR_LIBRARY_NAME "mesa" \
        --set LIBVA_DRIVERS_PATH "${pkgs.mesa}/lib/dri" \
        --prefix VK_ICD_FILENAMES : "${pkgs.mesa}/share/vulkan/icd.d/radeon_icd.x86_64.json:${pkgs.mesa}/share/vulkan/icd.d/intel_icd.x86_64.json:${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json" \
        --prefix VK_LAYER_PATH : "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d" \
        --set ALSA_PLUGIN_DIR "${pkgs.pipewire}/lib/alsa-lib"
      ''}
    '';

    # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
    # GCC 14 treats -Wincompatible-pointer-types as error by default
    NIX_CFLAGS_COMPILE = lib.optionalString pkgs.stdenv.isLinux "-Wno-error=incompatible-pointer-types";
    CARGO_PROFILE = "dev";
    CARGO_PROFILE_RELEASE_DEBUG = true;
  };
in {
  # Only export the binary - clippy and tests are run directly
  # via cargo in the development shell (nix develop)
  inherit bin;
}
