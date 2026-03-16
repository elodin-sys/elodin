{
  config,
  pkgs,
  rustToolchain,
  ...
}:
with pkgs; let
  # Import shared configuration
  common = pkgs.callPackage ./pkgs/common.nix {};
  iree_runtime = pkgs.callPackage ./pkgs/iree-runtime.nix {
    enableCuda = pkgs.stdenv.isLinux;
    enableMetal = pkgs.stdenv.isDarwin;
  };
  iree_runtime_tracy =
    if pkgs.stdenv.isLinux
    then
      pkgs.callPackage ./pkgs/iree-runtime.nix {
        enableCuda = true;
        enableTracing = true;
        tracySrc = fetchFromGitHub {
          owner = "wolfpld";
          repo = "tracy";
          rev = "5479a42ef9346b64e6d1b860ae58aa8abdb0c7f6";
          hash = "sha256-4J8b+72k+xpeT6KsrkioF1xfWEBsGg2eLRg9iONxP/I=";
        };
      }
    else null;
  llvm = llvmPackages_latest;

  # Base Python for use with venv (JAX 0.8+ and iree-base-compiler installed via pip)
  pythonBase = python313.withPackages (ps:
    with ps; [
      typing-extensions
      pytest
      pytest-json-report
      matplotlib
      polars
      numpy
    ]);
  shellAttrs = {
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
        pythonBase
        clang
        maturin
        bzip2
        libclang
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
        jq
        yq
        git
        git-filter-repo
        git-lfs

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
          # Tracy profiler (Linux-only: requires std::jthread, not in Apple libc++)
          tracy
          iree_runtime_tracy
          cudaPackages.cuda_cudart
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
      lib.optionals pkgs.stdenv.isDarwin [fixDarwinDylibNames]
    );

    # Environment variables
    LIBCLANG_PATH = "${libclang.lib}/lib";
    IREE_RUNTIME_DIR = "${iree_runtime}";
    IREE_RUNTIME_TRACY_DIR = lib.optionalString pkgs.stdenv.isLinux "${iree_runtime_tracy}";

    # The nox-py cdylib (.so) carries a DF_STATIC_TLS flag that forces glibc
    # to allocate ~10 KB from the tiny static-TLS surplus on dlopen.  Raise
    # the surplus so Python can import the extension without ENOMEM.
    GLIBC_TUNABLES = "glibc.rtld.optional_static_tls=16384";

    # GStreamer plugin path for elodinsink
    GST_PLUGIN_PATH = lib.makeSearchPathOutput "lib" "lib/gstreamer-1.0" [
      gst_all_1.gstreamer
      gst_all_1.gst-plugins-base
      gst_all_1.gst-plugins-good
      gst_all_1.gst-plugins-bad
      gst_all_1.gst-plugins-ugly
      config.packages.elodinsink
    ];

    LLDB_DEBUGSERVER_PATH = lib.optionalString pkgs.stdenv.isDarwin "/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Versions/A/Resources/debugserver";

    # Set up library paths for Linux graphics/audio
    LD_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isLinux (
      lib.makeLibraryPath [
        cudaPackages.cuda_cudart
      ]
      + ":"
      + common.makeLinuxLibraryPath {inherit pkgs;}
    );

    doCheck = false;
    shellHook = ''
      case "$(uname -s)" in
        Linux*)
          export CC=clang
          export CXX=clang++
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
      # start the shell if we're in an interactive shell
      if [[ $- == *i* ]]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 Elodin Development Shell (Nix)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "Environment ready:"
        echo "  • Rust: cargo, clippy, nextest"
        echo "  • Tools: uv, maturin, ruff, just, alejandra"
        echo ""
        echo "Development flow:"
        echo "  • Run 'just install' to build: elodin-py, elodin, and elodin-db"
        echo "  • don't forget to source the venv with 'source .venv/bin/activate'"
        echo ""

        exec ${pkgs.zsh}/bin/zsh
      fi
    '';
  };
  linuxShellAttrs = lib.optionalAttrs pkgs.stdenv.isLinux (
    common.linuxGraphicsEnv {inherit pkgs;}
    // {
      CC = "clang";
      CXX = "clang++";
      CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER = "clang";
      CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER = "clang";
    }
  );
in {
  # Unified shell that combines all development environments
  elodin = mkShell (shellAttrs // linuxShellAttrs);
}
