{ config, self', pkgs, lib, rustToolchain, ... }: {
  devShells = {
    rust = pkgs.mkShell {
      name = "elo-rust-shell";
      buildInputs = with pkgs;
        [
          buildkite-test-collector-rust
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
          openssl
          libclang
          gfortran
          gfortran.cc.lib
        ];
      LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
      doCheck = false;
    };
    elixir = pkgs.mkShell {
      name = "elo-elixir-shell";
      buildInputs = with pkgs;
        [
          elixir
          cacert
          esbuild
          tailwindcss
        ];
      doCheck = false;
      HEX_CACERTS_PATH = "/etc/ssl/certs/ca-bundle.crt";
      ESBUILD_BIN = "${pkgs.esbuild}/bin/esbuild";
      TAILWIND_BIN = "${pkgs.tailwindcss}/bin/tailwindcss";
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
          yq
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
    writing = pkgs.mkShell {
      name = "elo-writing-shell";
      buildInputs = with pkgs; [typos];
    };
  };
}
