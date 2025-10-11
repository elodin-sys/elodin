{
  config,
  pkgs,
  rustToolchain,
  ...
}:
with pkgs; let
  xla_ext = pkgs.callPackage ./pkgs/xla-ext.nix {system = pkgs.stdenv.hostPlatform.system;};
  llvm = llvmPackages_latest;
in {
  # Unified shell that combines all development environments
  elodin = mkShell {
    name = "elo-unified-shell";
    buildInputs =
      [
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
        python3
        python3Packages.jax
        python3Packages.jaxlib
        python3Packages.typing-extensions
        python3Packages.pytest
        python3Packages.pytest-json-report
        python3Packages.matplotlib
        python3Packages.polars
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
      ++ lib.optionals stdenv.isLinux [
        alsa-lib
        alsa-oss
        alsa-utils
        vulkan-loader
        wayland
        gtk3
        udev
        libxkbcommon
        fontconfig
        lldb
        autoPatchelfHook
        config.packages.elodin-py.py
      ]
      # macOS-specific dependencies
      ++ lib.optionals stdenv.isDarwin [
        fixDarwinDylibNames
      ];

    nativeBuildInputs = with pkgs; (
      lib.optionals stdenv.isLinux [autoPatchelfHook]
      ++ lib.optionals stdenv.isDarwin [fixDarwinDylibNames]
    );

    # Environment variables
    LIBCLANG_PATH = "${libclang.lib}/lib";
    XLA_EXTENSION_DIR = "${xla_ext}";

    # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
    # GCC 14 treats -Wincompatible-pointer-types as error by default
    NIX_CFLAGS_COMPILE = lib.optionalString stdenv.isLinux "-Wno-error=incompatible-pointer-types";

    LLDB_DEBUGSERVER_PATH = lib.optionalString stdenv.isDarwin "/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Versions/A/Resources/debugserver";

    doCheck = false;

    shellHook = ''
      set -euo pipefail

      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "🚀 Elodin Unified Development Shell"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo ""

      # Auto-setup venv and build elodin package for Python development
      VENV_DIR="$PWD/libs/nox-py/.venv"

      if [ -d "$PWD/libs/nox-py" ]; then
        if [ ! -d "$VENV_DIR" ]; then
          echo "🔨 First-time setup: Creating venv and building elodin..."
          (cd libs/nox-py && uv venv) 2>&1 | grep -v "arm64-apple-macosx" || true
          source "$VENV_DIR/bin/activate"
          (cd libs/nox-py && maturin develop) 2>&1 | grep -v "arm64-apple-macosx" || true
          echo "✅ Python setup complete!"
        else
          echo "📦 Using existing Python venv"
          source "$VENV_DIR/bin/activate"
        fi
      fi

      echo ""
      echo "🚀 Environment ready with:"
      echo "  • Shell: Your existing zsh configuration"
      echo "  • Enhanced tools: eza, bat, delta, fzf, ripgrep, zoxide"
      echo "  • Rust: cargo, clippy, nextest"
      echo "  • Python: uv, maturin, ruff (venv activated)"
      echo "  • Cloud: kubectl, gcloud, azure"
      echo "  • Version control: git with delta, git-lfs"
      echo ""
      echo "💡 All tools are available in your regular shell environment."
      echo "   Your existing zsh, Oh My Zsh, and p10k configs are preserved."
      echo ""

      # Start zsh if not already in it (uses user's default zsh config)
      if [ -z "''${ZSH_VERSION:-}" ]; then
        exec ${pkgs.zsh}/bin/zsh -l
      fi
    '';
  };
}
