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
        ];
      LIBCLANG_PATH = "${pkgs.llvmPackages_14.libclang.lib}/lib";
      BINDGEN_EXTRA_CLANG_ARGS = with pkgs; ''${lib.optionalString stdenv.cc.isGNU "-isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config} -idirafter ${stdenv.cc.cc}/lib/gcc/${stdenv.hostPlatform.config}/${lib.getVersion stdenv.cc.cc}/include"}'';
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
          nodejs_21
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
          (google-cloud-sdk.withExtraComponents (with google-cloud-sdk.components; [gke-gcloud-auth-plugin]))
        ];
      doCheck = false;
    };
  };
}
