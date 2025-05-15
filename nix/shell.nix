{
  config,
  pkgs,
  rustToolchain,
  ...
}: {
  c = (pkgs.mkShell.override {stdenv = pkgs.llvmPackages_19.libcxxStdenv;}) {
    name = "elo-c-shell";
    buildInputs = [];
    doCheck = false;
  };
  rust = pkgs.mkShell {
    name = "elo-rust-shell";
    buildInputs = with pkgs; [
      buildkite-test-collector-rust
      (rustToolchain pkgs)
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
      alsa-lib
      alsa-oss
      alsa-utils
      vulkan-loader
      wayland
      gtk3
      udev
      libxkbcommon
      fontconfig
      maturin
      cmake
      openssl
      libclang
      gfortran
      gfortran.cc.lib
      ffmpeg-full
      gst_all_1.gstreamer
      gst_all_1.gst-plugins-base
      gst_all_1.gst-plugins-good
    ];
    LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
    doCheck = false;
  };
  ops = pkgs.mkShell {
    name = "elo-ops-shell";
    buildInputs = with pkgs; [
      skopeo
      gettext
      just
      docker
      kubectl
      jq
      yq
      git-filter-repo
      (google-cloud-sdk.withExtraComponents (with google-cloud-sdk.components; [gke-gcloud-auth-plugin]))
      azure-cli
    ];
    doCheck = false;
  };
  python = pkgs.mkShell {
    name = "elo-py-shell";
    buildInputs = with pkgs; [
      ruff
      python3Packages.pytest
      python3Packages.pytest-json-report
      config.packages.elodin-py
    ];
  };

  nix-tools = pkgs.mkShell {
    name = "elo-nix-shell";
    buildInputs = with pkgs; [
      alejandra
    ];
  };
  writing = pkgs.mkShell {
    name = "elo-writing-shell";
    buildInputs = with pkgs; [typos];
  };
  docs = pkgs.mkShell {
    name = "elo-docs-shell";
    buildInputs = with pkgs; [
      typos
      zola
      ffmpeg
      rav1e
    ];
  };
}
