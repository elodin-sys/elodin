{
  config,
  pkgs,
  rustToolchain,
  ...
}:
with pkgs; let
  xla_ext = pkgs.callPackage ./pkgs/xla-ext.nix {system = pkgs.stdenv.hostPlatform.system;};
  llvm = llvmPackages_latest;

  # Create a Python environment with the same JAX version as our pyproject.toml
  pythonWithJax = let
    python3' = python3.override {
      packageOverrides = self: super: {
        # Override jaxlib to use version 0.4.31
        jaxlib = super.jaxlib-bin.overridePythonAttrs (old: rec {
          version = "0.4.31";
          src = let
            system = pkgs.stdenv.system;
            platform =
              if system == "x86_64-linux"
              then "manylinux2014_x86_64"
              else if system == "aarch64-linux"
              then "manylinux2014_aarch64"
              else "macosx_11_0_arm64";
            wheelName = "jaxlib-${version}-cp312-cp312-${platform}.whl";
            baseUrl = "https://files.pythonhosted.org/packages";
            wheelUrls = {
              "manylinux2014_x86_64" = "${baseUrl}/b1/09/58d35465d48c8bee1d9a4e7a3c5db2edaabfc7ac94f4576c9f8c51b83e70/${wheelName}";
              "manylinux2014_aarch64" = "${baseUrl}/e0/af/10b49f8de2acc7abc871478823579d7241be52ca0d6bb0d2b2c476cc1b68/${wheelName}";
              "macosx_11_0_arm64" = "${baseUrl}/68/cf/28895a4a89d88d18592507d7a35218b6bb2d8bced13615065c9f925f2ae1/${wheelName}";
            };
          in
            pkgs.fetchurl {
              url = wheelUrls.${platform} or (throw "Unsupported platform: ${platform}");
              hash =
                if system == "x86_64-linux"
                then "sha256-Hxr6X9WKYPZ/DKWG4mcUrs5i6qLIM0wk0OgoWvxKfM0="
                else if system == "aarch64-linux"
                then "sha256-TYZ6GgVlsxz9qrvsgeAwLGRhuyrEuSwEZwMo15WBmAM="
                else "sha256-yficGFKH5A7oFzpxQtZJUxHncs0TmpPcqT8NmcGHKDI="; # macosx_11_0_arm64
            };
        });
        jaxlib-bin = self.jaxlib;

        # Override JAX to use version 0.4.31
        jax = super.jax.overridePythonAttrs (old: rec {
          version = "0.4.31";
          src = super.fetchPypi {
            inherit (old) pname;
            inherit version;
            hash = "sha256-/S1HBkOgBz2CJzfweI9xORZWr35izFsueZXuOQzqwoc=";
          };
          pythonImportsCheck = [];
          doCheck = false;
        });
      };
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

      # Detect if we're in CI or non-interactive mode
      IS_CI="''${CI:-''${BUILDKITE:-''${GITHUB_ACTIONS:-}}}"
      IS_INTERACTIVE="''${PS1:-}"

      # Only show the banner in interactive mode
      if [ -n "$IS_INTERACTIVE" ] && [ -z "$IS_CI" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 Elodin Unified Development Shell"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
      fi

      # Auto-setup venv and build elodin package for Python development
      VENV_DIR="$PWD/libs/nox-py/.venv"

      if [ -d "$PWD/libs/nox-py" ]; then
        if [ ! -d "$VENV_DIR" ]; then
          if [ -n "$IS_INTERACTIVE" ] && [ -z "$IS_CI" ]; then
            echo "🔨 First-time setup: Creating venv and building elodin..."
          fi

          # Create venv
          if (cd libs/nox-py && uv venv) 2>&1 | grep -v "arm64-apple-macosx" || true; then
            # Try to build the package, but don't fail if it doesn't work
            if ! (cd libs/nox-py && source .venv/bin/activate && maturin develop) 2>&1 | grep -v "arm64-apple-macosx"; then
              if [ -n "$IS_CI" ]; then
                echo "⚠️  Warning: Failed to build elodin package (may be due to disk space or other issues)"
              else
                echo "⚠️  Warning: Failed to build elodin package - you may need to run 'maturin develop' manually"
              fi
            else
              [ -n "$IS_INTERACTIVE" ] && [ -z "$IS_CI" ] && echo "✅ Python setup complete!"
            fi
          fi
        else
          [ -n "$IS_INTERACTIVE" ] && [ -z "$IS_CI" ] && echo "📦 Using existing Python venv"
        fi

        # Activate the venv in the current shell if it exists
        if [ -f "$VENV_DIR/bin/activate" ]; then
          source "$VENV_DIR/bin/activate"
          [ -n "$IS_INTERACTIVE" ] && [ -z "$IS_CI" ] && echo "✅ Python venv activated: $VENV_DIR"

          # Export these so they persist if we launch zsh
          export VIRTUAL_ENV
          export PATH
        fi
      fi

      # Only show environment info and switch to zsh in interactive mode
      if [ -n "$IS_INTERACTIVE" ] && [ -z "$IS_CI" ]; then
        echo ""
        echo "🚀 Environment ready with:"
        echo "  • Shell: Your existing zsh configuration"
        echo "  • Enhanced tools: eza, bat, delta, fzf, ripgrep, zoxide"
        echo "  • Rust: cargo, clippy, nextest"
        echo "  • Python: uv, maturin, ruff (venv activated)"
        echo "  • Cloud: kubectl, gcloud, azure"
        echo "  • Version control: git with delta, git-lfs"
        echo ""

        # Only exec zsh if we're in an interactive shell and not already in zsh
        # and not in CI, and not in a nix develop --command context
        if [ -z "''${ZSH_VERSION:-}" ] && [ -z "''${IN_NIX_SHELL_COMMAND:-}" ]; then
          exec ${pkgs.zsh}/bin/zsh
        fi
      fi
    '';
  };
}
