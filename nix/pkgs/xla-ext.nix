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
    "aarch64-linux" = "sha256:vurmdMbrlRkfdnyF0hAdzorolYRNte+K4scEgIMf/+8=";
    "x86_64-linux" = "sha256:1zprdmrd597z5m4md7am50766n7sd3avsyn0yc0hkpywslp8mvbj";
  };
  suffix = builtins.getAttr system suffix_map;
  filename = "xla_extension-${version}-${system}-${suffix}";
  tarball = fetchTarball {
    url = "https://github.com/elodin-sys/xla-next/releases/download/v${version}/${filename}";
    sha256 = builtins.getAttr system sha256_map;
  };
in
  tarball
