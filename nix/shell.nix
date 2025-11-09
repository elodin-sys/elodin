{
  config,
  pkgs,
  rustToolchain,
  ...
}:
with pkgs; let
  # Import shared configuration
  common = pkgs.callPackage ./pkgs/common.nix {};
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
  elodin = mkShell (
    {
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
          # Use our custom Python with JAX 0.4.31
          pythonWithJax
          clang
          maturin
          bzip2
          libclang
          sccache
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
        ++ common.commonNativeBuildInputs
        ++ common.commonBuildInputs
        # Linux-specific dependencies
        ++ lib.optionals pkgs.stdenv.isLinux (
          common.linuxGraphicsAudioDeps
          ++ [
            # Additional Linux-specific tools not in common
            alsa-oss
            alsa-utils
            gtk3
            fontconfig
            lldb
            autoPatchelfHook
            config.packages.elodin-py.py
          ]
        )
        # macOS-specific dependencies
        ++ lib.optionals pkgs.stdenv.isDarwin (
          common.darwinDeps
          ++ [
            fixDarwinDylibNames
          ]
        );

      nativeBuildInputs = with pkgs; (
        lib.optionals pkgs.stdenv.isLinux [autoPatchelfHook]
        ++ lib.optionals pkgs.stdenv.isDarwin [fixDarwinDylibNames]
      );

      # Environment variables
      LIBCLANG_PATH = "${libclang.lib}/lib";
      XLA_EXTENSION_DIR = "${xla_ext}";

      # Enable sccache for local nix develop shell only for faster Rust builds
      # SCCACHE_BIN = "${pkgs.sccache}/bin/sccache";

      # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
      # GCC 14 treats -Wincompatible-pointer-types as error by default
      NIX_CFLAGS_COMPILE = common.netlibWorkaround;

      LLDB_DEBUGSERVER_PATH = lib.optionalString pkgs.stdenv.isDarwin "/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Versions/A/Resources/debugserver";

      # Set up library paths for Linux graphics/audio
      LD_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isLinux (
        common.makeLinuxLibraryPath {inherit pkgs;}
      );

      doCheck = false;

      shellHook = ''
        # start the shell if we're in an interactive shell
        if [[ $- == *i* ]]; then
          echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
          echo "🚀 Elodin Development Shell (Nix)"
          echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
          echo ""
          echo "Environment ready:"
          echo "  • Rust: cargo, clippy, nextest"
          echo "  • Tools: uv, maturin, ruff, just, kubectl, gcloud"
          echo "  • Shell tools: eza, bat, delta, fzf, ripgrep, zoxide"
          echo ""
          echo "SDK Development (if needed):"
          echo "  "
          echo "uv venv --python 3.12"
          echo "source .venv/bin/activate"
          echo "uvx maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml"
          echo ""

          export RUSTC_WRAPPER=""
          # export RUSTC_WRAPPER="''${SCCACHE_BIN}"
          # export SCCACHE_SERVER_PORT=4226
          # export SCCACHE_DIR="/tmp/sc"
          exec ${pkgs.zsh}/bin/zsh
        fi
      '';
    }
    // lib.optionalAttrs pkgs.stdenv.isLinux (
      common.linuxGraphicsEnv {inherit pkgs;}
    )
  );
}
