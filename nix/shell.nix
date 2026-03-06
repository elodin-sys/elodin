{
  config,
  pkgs,
  rustToolchain,
  ...
}:
with pkgs; let
  # Import shared configuration
  common = pkgs.callPackage ./pkgs/common.nix {};
  iree_runtime = pkgs.callPackage ./pkgs/iree-runtime.nix {};
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

        # Tracy profiler
        tracy
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
      lib.optionals pkgs.stdenv.isDarwin [fixDarwinDylibNames]
    );

    # Environment variables
    LIBCLANG_PATH = "${libclang.lib}/lib";
    IREE_RUNTIME_DIR = "${iree_runtime}";

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
      export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
      if [ -f "$REPO_ROOT/nix/shellrc" ]; then
        export NIX_SHELLRC="$REPO_ROOT/nix/shellrc"
      fi
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
        echo "  • Run 'install-elodin' to provide: elodin-py, elodin, and elodin-db"
        # NIX_SHELLRC_READY promises the shell will consume the rc file. We can't test that it
        # has happened because it won't happen until the new zsh is exec'd. And we can't ensure that
        # it did happen once exec'd unless we could run commands in that shell--which is exactly the
        # kind of facility that we get by asking the user to source "$NIX_SHELLRC".
        if [ "''${NIX_SHELLRC_READY:-0}" -ne 1 ]; then
            echo "WARNING: No 'install-elodin' shell function available. Run this to fix:" >&2;
            echo ""
            echo 'source $NIX_SHELLRC;' >&2;
            echo ""
            echo "Or add this to your .zshrc to fix permanently:" >&2;
            echo 'export NIX_SHELLRC_READY=1; [[ -n "$NIX_SHELLRC" ]] && source "$NIX_SHELLRC"' >&2;
        fi
        echo ""

        # HOOK

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
}
