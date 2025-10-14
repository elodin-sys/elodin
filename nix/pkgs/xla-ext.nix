{
  pkgs,
  system,
  lib,
  ...
}: let
  version = "0.9.1";
  suffix_map = {
    "aarch64-darwin" = "cpu.tar.gz";
    "aarch64-linux" = "gnu-cpu.tar.gz";
    "x86_64-linux" = "gnu-cpu.tar.gz";
  };
  sha256_map = {
    "aarch64-darwin" = "sha256:0m82waljhscajsdkpyd16c2spn1avfpp9a9am9nahlafa6lvrvrz";
    "aarch64-linux" = "sha256:0x88bffhqprnjagf8j6jgdydhp4z7c383z9m8m93ns7z93y5iw67";
    "x86_64-linux" = "sha256:11j0fy1p8wc79pgdpym5dn9k6w65n8jja8mqnfyqpmhkppl0idg6";
  };
  suffix = builtins.getAttr system suffix_map;
  filename = "xla_extension-${version}-${system}-${suffix}";
  tarball = fetchTarball {
    url = "https://github.com/elodin-sys/xla-next/releases/download/v${version}/${filename}";
    sha256 = builtins.getAttr system sha256_map;
  };
in
  tarball
