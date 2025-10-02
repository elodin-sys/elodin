{
  config,
  pkgs,
  rustToolchain,
  ...
}:
with pkgs; let
  xla_ext = pkgs.callPackage ./pkgs/xla-ext.nix {};
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
        libclang
        gfortran
        gfortran.cc.lib
        ffmpeg-full
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
    buildInputs = [
      ruff
      cmake
      python3Packages.pytest
      python3Packages.pytest-json-report
      config.packages.elodin-py.py
    ];
    nativeBuildInputs = with pkgs; (
      lib.optionals stdenv.isLinux [autoPatchelfHook]
      ++ lib.optionals stdenv.isDarwin [fixDarwinDylibNames]
    );
    XLA_EXTENSION_DIR = "${xla_ext}";
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
}
