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
  shellAttrs = {
    # Unified shell that combines all development environments
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
        # Ensure gfortran is available and prioritized for netlib-src builds
        gfortran
        ffmpeg-full
        ffmpeg-full.dev
        gst_all_1.gstreamer
        gst_all_1.gst-plugins-base
        gst_all_1.gst-plugins-good
        gst_all_1.gst-plugins-bad # For h264parse
        gst_all_1.gst-plugins-ugly # For x264enc (H.264 encoding)
        config.packages.elodinsink # GStreamer plugin for Elodin-DB video streaming
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

    # GStreamer plugin path for elodinsink
    GST_PLUGIN_PATH = lib.makeSearchPathOutput "lib" "lib/gstreamer-1.0" [
      gst_all_1.gstreamer
      gst_all_1.gst-plugins-base
      gst_all_1.gst-plugins-good
      gst_all_1.gst-plugins-bad
      gst_all_1.gst-plugins-ugly
      config.packages.elodinsink
    ];

    # Workaround for netlib-src 0.8.0 incompatibility with GCC 14+
    # GCC 14 treats -Wincompatible-pointer-types as error by default
    NIX_CFLAGS_COMPILE = common.netlibWorkaround;

    # Force CMake to use Nix Fortran compiler instead of system one
    # This fixes netlib-src build failures when system Fortran is incompatible
    FC = "${pkgs.gfortran}/bin/gfortran";
    F77 = "${pkgs.gfortran}/bin/gfortran";
    # Explicitly tell CMake which Fortran compiler to use
    CMAKE_Fortran_COMPILER = "${pkgs.gfortran}/bin/gfortran";

    LLDB_DEBUGSERVER_PATH = lib.optionalString pkgs.stdenv.isDarwin "/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Versions/A/Resources/debugserver";

    # Set up library paths for Linux graphics/audio
    LD_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isLinux (
      common.makeLinuxLibraryPath {inherit pkgs;}
    );

    doCheck = false;

    shellHook = ''
      case "$(uname -s)" in
        Linux*)
          # Use existing DISPLAY if set, otherwise default to :0
          if [ -z "$DISPLAY" ]; then
            export DISPLAY=:0
          fi
          export WINIT_UNIX_BACKEND=x11
          unset WAYLAND_DISPLAY
          export XDG_SESSION_TYPE=x11
          # Ensure X11 libraries are available
          export LD_LIBRARY_PATH="${lib.makeLibraryPath (with pkgs; [
        xorg.libX11
        xorg.libXcursor
        xorg.libXrandr
        xorg.libXi
        xorg.libXext
        libxkbcommon
        mesa
        libGL
      ])}:''${LD_LIBRARY_PATH}"
        ;;
      esac

      # Ensure Nix gfortran is in PATH before system Fortran
      # This is critical for netlib-src builds
      export PATH="${pkgs.gfortran}/bin:''${PATH}"

      # Explicitly set Fortran compiler for CMake
      export FC="${pkgs.gfortran}/bin/gfortran"
      export F77="${pkgs.gfortran}/bin/gfortran"
      export CMAKE_Fortran_COMPILER="${pkgs.gfortran}/bin/gfortran"

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

        exec ${pkgs.zsh}/bin/zsh
      fi
    '';
  };
  linuxShellAttrs = lib.optionalAttrs pkgs.stdenv.isLinux (
    common.linuxGraphicsEnv {inherit pkgs;}
  );
in {
  # Unified shell that combines all development environments
  elodin = mkShell (shellAttrs // linuxShellAttrs);

  # Quick shell reuses the default shell machinery but skips heavy Rust package builds.
  quick = mkShell (
    (shellAttrs
      // {
        name = "elo-quick-shell";
        buildInputs = lib.subtractLists [
          config.packages.elodinsink
          config.packages.elodin-py.py
        ] shellAttrs.buildInputs;
        GST_PLUGIN_PATH = lib.makeSearchPathOutput "lib" "lib/gstreamer-1.0" [
          gst_all_1.gstreamer
          gst_all_1.gst-plugins-base
          gst_all_1.gst-plugins-good
          gst_all_1.gst-plugins-bad
          gst_all_1.gst-plugins-ugly
        ];
      })
    // linuxShellAttrs
  );

  # Profiling shell adds Tracy GUI without duplicating the base shell definition
  elodin-profiling = mkShell (
    (shellAttrs
      // {
        name = "elo-profiling-shell";
        buildInputs = shellAttrs.buildInputs ++ [tracy];
        shellHook =
          lib.replaceStrings
          [
            "exec ${pkgs.zsh}/bin/zsh"
          ]
          [
            ''
              echo "Profiling shell:"
              echo "  • Tracy GUI is available by running 'tracy'"
              echo "  • Run Elodin with: RUST_LOG=info cargo run --release -p elodin --features tracy"
              echo ""

              exec ${pkgs.zsh}/bin/zsh
            ''
          ]
          shellAttrs.shellHook;
      })
    // linuxShellAttrs
  );
}
