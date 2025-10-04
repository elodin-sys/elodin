{
  pkgs,
  system,
  lib,
  ...
}: let
  version = "0.5.4";
  suffix_map = {
    "aarch64-darwin" = "cpu.tar.gz";
    "aarch64-linux" = "gnu-cpu.tar.gz";
    "x86_64-linux" = "gnu-cpu.tar.gz";
  };
  sha256_map = {
    "aarch64-darwin" = "sha256:0ykfnp6d78vp2yrhmr8wa3rlv6cri6mdl0fg034za839j7i7xqkz";
    "aarch64-linux" = "sha256:0sy53r6qhw0n3n342s013nq5rnzlg1qdbmgpvawh3p35a21qy8xr";
    "x86_64-linux" = "sha256:103mybbnz6fm2i3r0fy0nf23ffdjxb37wd4pzvmwn0dpczr6dkw1";
  };
  suffix = builtins.getAttr system suffix_map;
  filename = "xla_extension-${system}-${suffix}";
  tarball = fetchTarball {
    url = "https://github.com/elodin-sys/xla/releases/download/v${version}/${filename}";
    sha256 = builtins.getAttr system sha256_map;
  };
in
  tarball
