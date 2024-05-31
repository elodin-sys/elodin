{ config, self', pkgs, lib, rustToolchain, ... }: {
  devShells = {
    rust = pkgs.mkShell {
      name = "elo-rust-shell";
      buildInputs = with pkgs;
        [
          config.packages.buildkite-test-collector
          rustToolchain
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
          protobuf
          sccache
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
          openblas
          lapack
          openssl
          libclang
        ];
      LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
      doCheck = false;
    };
    elixir = pkgs.mkShell {
      name = "elo-elixir-shell";
      buildInputs = with pkgs;
        [
          elixir
        ];
      doCheck = false;
    };
    node = pkgs.mkShell.override {stdenv = pkgs.gcc12Stdenv;} {
      name = "elo-node-shell";
      buildInputs = with pkgs;
        [
          nodejs_22
        ];
      doCheck = false;
    };
    ops = pkgs.mkShell {
      name = "elo-ops-shell";
      buildInputs = with pkgs;
        [
          _1password
          skopeo
          gettext
          just
          docker
          kubectl
          jq
          git-filter-repo
          (google-cloud-sdk.withExtraComponents (with google-cloud-sdk.components; [gke-gcloud-auth-plugin]))
        ];
      doCheck = false;
    };
    python = pkgs.mkShell {
      name = "elo-py-shell";
      buildInputs = with pkgs;
        [
          ruff
          python3Packages.pytest
          python3Packages.pytest-json-report
          config.packages.elodin-py
        ];
    };
  };
}
