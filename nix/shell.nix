{
  config,
  pkgs,
  rustToolchain,
  ...
}:
with pkgs; let
  # Import shared configuration
  common = pkgs.callPackage ./pkgs/common.nix {};
  llvm = llvmPackages_latest;

  # Base Python for use with venv (JAX 0.8+ installed via pip)
  pythonBase = python313.withPackages (ps:
    with ps; [
      typing-extensions
      pytest
      pytest-json-report
      matplotlib
      polars
      numpy
      grpcio
      grpcio-tools
      grpcio-health-checking
      grpcio-reflection
      pyarrow
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
        vim # Editor
        less # Pager

        # Fonts for terminal (MesloLGS for p10k)
        (nerd-fonts.meslo-lg)

        # Rust toolchain and tools
        buildkite-test-collector-rust
        (rustToolchain pkgs)
        cargo-nextest
        protobuf
        grpc
        pythonBase
        clang
        maturin
        bzip2
        libclang
        gfortran
        ffmpeg-full
        ffmpeg-full.dev
        common.ktxTools # Provides toktx for generated skybox KTX2 cubemaps
        gst_all_1.gstreamer
        gst_all_1.gst-plugins-base
        gst_all_1.gst-plugins-good
        gst_all_1.gst-plugins-bad # For h264parse and hardware encoders
        gst_all_1.gst-plugins-ugly # For portable x264 H.264 encoding
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
        time

        # Documentation and quality tools
        alejandra
        buf
        typos
        zola
        rav1e
      ]
      ++ common.commonNativeBuildInputs
      ++ common.commonBuildInputs
      # Linux-specific dependencies
      ++ lib.optionals pkgs.stdenv.isLinux (
        common.linuxGraphicsAudioDeps
        ++ common.linuxCaptureTools
        ++ [
          # Additional Linux-specific tools not in common
          iproute2
          tcpdump
          alsa-oss
          alsa-utils
          gtk3
          fontconfig
          lldb
          autoPatchelfHook
          # Tracy profiler (Linux-only: requires std::jthread, not in Apple libc++)
          tracy
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
    TOKTX = "${common.ktxTools}/bin/toktx";

    # The nox-py cdylib (.so) carries a DF_STATIC_TLS flag that forces glibc
    # to allocate ~10 KB from the tiny static-TLS surplus on dlopen.  Raise
    # the surplus so Python can import the extension without ENOMEM.
    GLIBC_TUNABLES = "glibc.rtld.optional_static_tls=16384";

    # GStreamer + elodinsink for video-stream and capture. Example scripts
    # inherit this; do not prepend cargo `target/` (see makeGstPluginPath).
    GST_PLUGIN_PATH = common.makeGstPluginPath {
      inherit pkgs;
      extra = [config.packages.elodinsink];
    };

    LLDB_DEBUGSERVER_PATH = lib.optionalString pkgs.stdenv.isDarwin "/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Versions/A/Resources/debugserver";

    UV_PYTHON = "${pythonBase}/bin/python3";

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
          ${common.linuxEditorShellHook}
        ;;
      esac

      export ELODIN_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

      if [ -z "''${ELODIN_SHELL_ID:-}" ]; then
        if [ "$(uname -s)" = Linux ]; then
          ELODIN_SHELL_ID="$(ps -o sid= -p $$ | tr -d '[:space:]')"
        else
          ELODIN_SHELL_ID="$$"
        fi
      fi
      export ELODIN_SHELL_ID
      _elo_shell_dir="$ELODIN_ROOT/target/shells/$ELODIN_SHELL_ID"
      export ELODIN_SHELL_BIN="$_elo_shell_dir/bin"
      mkdir -p "$ELODIN_SHELL_BIN"
      export VIRTUAL_ENV="$_elo_shell_dir/venv"
      export UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV"
      if [ ! -x "$VIRTUAL_ENV/bin/python" ]; then
        uv venv --quiet --python 3.13 --python-preference only-system --allow-existing "$VIRTUAL_ENV"
      fi
      PATH="$ELODIN_SHELL_BIN:$VIRTUAL_ENV/bin:$PATH"
      export UV_PYTHON="$VIRTUAL_ENV/bin/python"
      export PATH

      if [ -d "$ELODIN_ROOT/target/shells" ]; then
        for _elo_dir in "$ELODIN_ROOT/target/shells"/[0-9]*; do
          [ -d "$_elo_dir" ] || continue
          _elo_id="''${_elo_dir##*/}"
          case "$_elo_id" in
            *[!0-9]*) continue ;;
          esac
          if ! kill -0 "$_elo_id" 2>/dev/null; then
            rm -rf "$_elo_dir"
          fi
        done
        unset _elo_dir _elo_id
      fi

      alias zar='gtar --zstd --sparse'

      # nix develop applies this hook via print-dev-env (non-interactive) then
      # starts $SHELL. Set SHELL/ZDOTDIR here so -c env and interactive agree.
      if [ "''${ELODIN_SHELL:-}" = zsh ]; then
        export SHELL=${pkgs.zsh}/bin/zsh
        export ELODIN_NIX_PATH="$PATH"
        export ZDOTDIR="$_elo_shell_dir/zdot"
        mkdir -p "$ZDOTDIR"
        printf '%s\n' \
          'ZDOTDIR="$HOME"' \
          '[ -f "$HOME/.zshenv" ] && . "$HOME/.zshenv"' \
          '[ -f "$HOME/.zshrc" ] && . "$HOME/.zshrc"' \
          'typeset -U path' \
          'path=(''${(s.:.)ELODIN_NIX_PATH} $path)' \
          "alias zar='gtar --zstd --sparse'" \
          > "$ZDOTDIR/.zshrc"
      fi
      unset _elo_shell_dir

      if [[ $- == *i* ]]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 Elodin Development Shell (Nix)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "Environment ready:"
        echo "  • Rust: cargo, clippy, nextest"
        echo "  • Tools: uv, maturin, ruff, just, alejandra"
        echo "  • Venv: $VIRTUAL_ENV (auto-active; no source needed)"
        echo "  • Local bins: $ELODIN_SHELL_BIN"
        echo ""
        echo "Development flow:"
        echo "  • just local-install  — this shell only (use this in agents / parallel worktrees)"
        echo "  • just install        — global ~/.cargo/bin (shared across shells)"
        echo "  • ELODIN_SHELL=zsh    — opt-in zsh + p10k (PATH is re-asserted after ~/.zshrc)"
        echo ""

        if [ "''${ELODIN_SHELL:-}" = zsh ]; then
          exec "$SHELL" -i
        fi
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
