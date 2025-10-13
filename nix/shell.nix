{
  config,
  pkgs,
  rustToolchain,
  ...
}:
with pkgs; let
  xla_ext = pkgs.callPackage ./pkgs/xla-ext.nix {system = pkgs.stdenv.hostPlatform.system;};
  llvm = llvmPackages_latest;

  # Import shared JAX overrides
  jaxOverrides = pkgs.callPackage ./pkgs/jax-overrides.nix {inherit pkgs;};

  # Create a Python environment with the same JAX version as our pyproject.toml
  pythonWithJax = let
    python3' = python3.override {
      packageOverrides = jaxOverrides;
    };
  in
    python3'.withPackages (ps:
      with ps; [
        jax
        jaxlib
        typing-extensions
        pytest
        pytest-json-report
        matplotlib
        polars
        numpy
      ]);
in {
  # Unified shell that combines all development environments
  elodin = mkShell {
    name = "elo-unified-shell";
    buildInputs =
      [
        # Interactive bash (required for nix develop to work properly)
        bashInteractive

        # Shell stack
        zsh
        oh-my-zsh
        zsh-powerlevel10k
        zsh-completions

        # Enhanced CLI tools
        eza # Better ls
        bat # Better cat
        delta # Better git diff
        fzf # Fuzzy finder
        fd # Better find
        ripgrep # Better grep
        zoxide # Smart cd
        direnv # Directory environments
        nix-direnv # Nix integration for direnv
        vim # Editor
        less # Pager

        # Fonts for terminal (MesloLGS for p10k)
        (nerd-fonts.meslo-lg)

        # Rust toolchain and tools
        buildkite-test-collector-rust
        (rustToolchain pkgs)
        cargo-nextest
        pkg-config
        # Use our custom Python with JAX 0.4.31
        pythonWithJax
        openssl
        clang
        maturin
        cmake
        openssl
        xz
        bzip2
        libclang
        gfortran
        gfortran.cc.lib
        ffmpeg-full
        ffmpeg-full.dev
        gst_all_1.gstreamer
        gst_all_1.gst-plugins-base
        gst_all_1.gst-plugins-good
        flip-link

        # Python tools
        ruff
        uv

        # Operations tools
        skopeo
        gettext
        just
        docker
        kubectl
        jq
        yq
        git-filter-repo
        git-lfs
        (google-cloud-sdk.withExtraComponents (
          with google-cloud-sdk.components; [gke-gcloud-auth-plugin]
        ))
        azure-cli

        # Documentation and quality tools
        alejandra
        typos
        zola
        rav1e
      ]
      # Linux-specific dependencies
      ++ lib.optionals pkgs.stdenv.isLinux [
        # Audio
        alsa-lib
        alsa-oss
        alsa-utils
        pipewire  # For PipeWire support
        
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
        xorg.libX11
        xorg.libXcursor
        xorg.libXrandr
        xorg.libXi
        xorg.libXext
        xorg.libxshmfence
        
        # Wayland
        wayland
        libxkbcommon
        
        # Other
        gtk3
        udev
        systemd  # For libudev
        fontconfig
        lldb
        autoPatchelfHook
        config.packages.elodin-py.py
        
        # Additional build dependencies from CI
        openblas
      ]
      # macOS-specific dependencies
      ++ lib.optionals pkgs.stdenv.isDarwin [
        fixDarwinDylibNames
      ];

    nativeBuildInputs = with pkgs; (
      lib.optionals pkgs.stdenv.isLinux [autoPatchelfHook]
      ++ lib.optionals pkgs.stdenv.isDarwin [fixDarwinDylibNames]
    );

    # Environment variables
    LIBCLANG_PATH = "${libclang.lib}/lib";
    XLA_EXTENSION_DIR = "${xla_ext}";

    # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
    # GCC 14 treats -Wincompatible-pointer-types as error by default
    NIX_CFLAGS_COMPILE = lib.optionalString pkgs.stdenv.isLinux "-Wno-error=incompatible-pointer-types";

    LLDB_DEBUGSERVER_PATH = lib.optionalString pkgs.stdenv.isDarwin "/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Versions/A/Resources/debugserver";
    
    # Set up library paths for Linux graphics/audio
    LD_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isLinux (
      lib.makeLibraryPath [
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
        xorg.libX11
        xorg.libXcursor
        xorg.libXrandr
        xorg.libXi
        xorg.libXext
        xorg.libxshmfence
        
        # Wayland
        wayland
        libxkbcommon
        
        # Other
        udev
        systemd
      ]
    );
    
    # Graphics environment variables for Linux
    LIBGL_DRIVERS_PATH = lib.optionalString pkgs.stdenv.isLinux "${mesa}/lib/dri";
    __GLX_VENDOR_LIBRARY_NAME = lib.optionalString pkgs.stdenv.isLinux "mesa";
    LIBVA_DRIVERS_PATH = lib.optionalString pkgs.stdenv.isLinux "${mesa}/lib/dri";
    VK_ICD_FILENAMES = lib.optionalString pkgs.stdenv.isLinux "${mesa}/share/vulkan/icd.d/radeon_icd.x86_64.json:${mesa}/share/vulkan/icd.d/intel_icd.x86_64.json:${mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json";
    VK_LAYER_PATH = lib.optionalString pkgs.stdenv.isLinux "${vulkan-validation-layers}/share/vulkan/explicit_layer.d";
    ALSA_PLUGIN_DIR = lib.optionalString pkgs.stdenv.isLinux "${pipewire}/lib/alsa-lib";


    doCheck = false;

    shellHook = ''
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "🚀 Elodin Development Shell (Nix)"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo ""
      echo "Environment ready:"
      echo "  • Rust: cargo, clippy, nextest"
      echo "  • Tools: uv, maturin, ruff, just, kubectl, gcloud"
      echo "  • Shell tools: eza, bat, delta, fzf, ripgrep, zoxide"
      echo ""
      echo "💡 Python setup (if needed):"
      echo "   cd libs/nox-py && uv venv --python 3.12"
      echo "   source .venv/bin/activate && uvx maturin develop --uv"
      echo ""

      # If we're in an interactive shell and not already in zsh, exec into zsh
      if [[ $- == *i* ]] && [ -z "''${ZSH_VERSION:-}" ]; then
        exec ${pkgs.zsh}/bin/zsh
      fi
    '';
  };
}
