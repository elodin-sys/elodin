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
  c = (mkShell.override {stdenv = llvm.libcxxStdenv;}) {
    name = "elo-c-shell";
    buildInputs = [];
    doCheck = false;
  };
  rust = mkShell {
    name = "elo-rust-shell";
    buildInputs =
      [
        buildkite-test-collector-rust
        (rustToolchain pkgs)
        cargo-nextest
        pkg-config
        python3
        python3Packages.jax
        python3Packages.jaxlib
        python3Packages.typing-extensions
        python3Packages.pytest
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
      ]
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
      ];
    LIBCLANG_PATH = "${libclang.lib}/lib";
    doCheck = false;

    # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
    # GCC 14 treats -Wincompatible-pointer-types as error by default
    NIX_CFLAGS_COMPILE = lib.optionalString stdenv.isLinux "-Wno-error=incompatible-pointer-types";

    LLDB_DEBUGSERVER_PATH = lib.optionalString stdenv.isDarwin "/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Versions/A/Resources/debugserver";
    XLA_EXTENSION_DIR = "${xla_ext}";
  };
  ops = mkShell {
    name = "elo-ops-shell";
    buildInputs = [
      skopeo
      gettext
      just
      docker
      kubectl
      jq
      yq
      git-filter-repo
      (google-cloud-sdk.withExtraComponents (
        with google-cloud-sdk.components; [gke-gcloud-auth-plugin]
      ))
      azure-cli
    ];
    doCheck = false;
  };
  python = mkShell {
    name = "elo-py-shell";
    buildInputs =
      [
        ruff
        cmake
        python3Packages.pytest
        python3Packages.pytest-json-report
      ]
      # On Linux: use pre-built wheel via crane
      # On macOS: crane has path bugs, so provide maturin for manual build
      ++ lib.optionals stdenv.isLinux [
        config.packages.elodin-py.py
      ]
      ++ lib.optionals stdenv.isDarwin [
        maturin
        uv
        (rustToolchain pkgs)
        pkg-config
        gfortran
        gfortran.cc.lib
        bzip2
        xz
        openssl
        libclang
      ];
    nativeBuildInputs = with pkgs; (
      lib.optionals stdenv.isLinux [autoPatchelfHook]
      ++ lib.optionals stdenv.isDarwin [fixDarwinDylibNames]
    );
    XLA_EXTENSION_DIR = "${xla_ext}";
    LIBCLANG_PATH = lib.optionalString stdenv.isDarwin "${libclang.lib}/lib";

    # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
    # GCC 14 treats -Wincompatible-pointer-types as error by default
    NIX_CFLAGS_COMPILE = lib.optionalString stdenv.isLinux "-Wno-error=incompatible-pointer-types";

    shellHook = lib.optionalString stdenv.isDarwin ''
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "📦 macOS Python Development Shell"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo ""

      # Auto-setup venv and build elodin package
      VENV_DIR="$PWD/libs/nox-py/.venv"

      if [ ! -d "$VENV_DIR" ]; then
        echo "🔨 First-time setup: Creating venv and building elodin..."
        (cd libs/nox-py && uv venv) 2>&1 | grep -v "arm64-apple-macosx" || true
        source "$VENV_DIR/bin/activate"
        (cd libs/nox-py && maturin develop) 2>&1 | grep -v "arm64-apple-macosx" || true
        echo ""
        echo "✅ Setup complete!"
      else
        echo "📦 Using existing venv (run 'rm -rf libs/nox-py/.venv' to rebuild)"
        source "$VENV_DIR/bin/activate"
      fi

      echo ""
      echo "ℹ️  Python ready! Test with: python -c 'import elodin; print(elodin.__version__)'"
      echo ""
    '';
  };

  nix-tools = mkShell {
    name = "elo-nix-shell";
    buildInputs = [
      alejandra
    ];
  };
  writing = mkShell {
    name = "elo-writing-shell";
    buildInputs = [typos];
  };
  docs = mkShell {
    name = "elo-docs-shell";
    buildInputs = [
      typos
      zola
      ffmpeg
      rav1e
    ];
  };

  # Unified shell that combines all development environments
  elodin = mkShell {
    name = "elo-unified-shell";
    buildInputs =
      [
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

    shellHook = lib.optionalString stdenv.isDarwin ''
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "🚀 Elodin Unified Development Shell"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo ""
      echo "This shell includes tools for:"
      echo "  • Rust development (cargo, clippy, nextest)"
      echo "  • Python development (uv, maturin, ruff)"
      echo "  • C/C++ compilation"
      echo "  • Cloud operations (kubectl, gcloud, azure)"
      echo "  • Documentation (zola, typos)"
      echo ""

      # Auto-setup venv and build elodin package for Python development
      VENV_DIR="$PWD/libs/nox-py/.venv"

      if [ -d "$PWD/libs/nox-py" ]; then
        if [ ! -d "$VENV_DIR" ]; then
          echo "🔨 First-time setup: Creating venv and building elodin..."
          (cd libs/nox-py && uv venv) 2>&1 | grep -v "arm64-apple-macosx" || true
          source "$VENV_DIR/bin/activate"
          (cd libs/nox-py && maturin develop) 2>&1 | grep -v "arm64-apple-macosx" || true
          echo ""
          echo "✅ Python setup complete!"
        else
          echo "📦 Using existing venv (run 'rm -rf libs/nox-py/.venv' to rebuild)"
          source "$VENV_DIR/bin/activate"
        fi

        echo ""
        echo "ℹ️  Python ready! Test with: python -c 'import elodin; print(elodin.__version__)'"
      fi

      echo ""
    '';
  };
}
