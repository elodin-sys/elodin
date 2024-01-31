{ config, self', pkgs, lib, flakeInputs, rustToolchain, system, ... }:
let
  xla_sha256_map = {
    "aarch64-darwin" = "sha256:0ykfnp6d78vp2yrhmr8wa3rlv6cri6mdl0fg034za839j7i7xqkz";
    "aarch64-linux" = "sha256:0sy53r6qhw0n3n342s013nq5rnzlg1qdbmgpvawh3p35a21qy8xr";
    "x86_64-linux"   = "sha256:103mybbnz6fm2i3r0fy0nf23ffdjxb37wd4pzvmwn0dpczr6dkw1";
  };
  xla_ext = fetchTarball {
    url = "https://github.com/elodin-sys/xla/releases/download/v0.5.4/xla_extension-${system}-gnu-cpu.tar.gz";
    sha256 = builtins.getAttr system xla_sha256_map;
  };
  craneLib = (flakeInputs.crane.mkLib pkgs).overrideToolchain rustToolchain;
  crateName = craneLib.crateNameFromCargoToml { cargoToml = ../services/sim-agent/Cargo.toml; };
  src = pkgs.nix-gitignore.gitignoreSource [] ../.;
  commonArgs = {
    inherit (crateName) pname version;
    inherit src;
    doCheck = false;
    cargoExtraArgs = "--package=${crateName.pname}";
    buildInputs = with pkgs;
      [
        systemdMinimal
        alsa-lib
        pkg-config
        (python3.withPackages (ps: with ps; [numpy jax jaxlib]))
        clang
        protobuf
        pango
        gtk3
      ]
      ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
        pkgs.libiconv
      ];
      XLA_EXTENSION_DIR = "${xla_ext}";
      LIBCLANG_PATH = "${pkgs.llvmPackages_14.libclang.lib}/lib";
      BINDGEN_EXTRA_CLANG_ARGS = with pkgs; ''${lib.optionalString stdenv.cc.isGNU "-isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config} -idirafter ${stdenv.cc.cc}/lib/gcc/${stdenv.hostPlatform.config}/${lib.getVersion stdenv.cc.cc}/include"}'';
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  bin = craneLib.buildPackage (commonArgs // {
    inherit cargoArtifacts;
  });

  image = pkgs.dockerTools.buildLayeredImage {
    name = "elo-sim-agent";
    tag = "latest";
    contents = with pkgs; [
      cacert
      busybox
      (python3.withPackages (ps: with ps; [numpy jax jaxlib]))
    ];
    config = {
      Env = ["SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"];
      Cmd = ["${bin}/bin/${crateName.pname}"];
    };
  };
in
{
  packages.sim-agent-image = image;
}
