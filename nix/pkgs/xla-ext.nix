{
  pkgs,
  system,
  lib,
  ...
}: let
  xla_ext_ver = "0.9.1";
  xla_suffix_map = {
    "aarch64-darwin" = "cpu.tar.gz";
    "aarch64-linux" = "gnu-cpu.tar.gz";
    "x86_64-linux" = "gnu-cpu.tar.gz";
  };
  xla_sha256_map = {
    "aarch64-darwin" = "sha256:0hv7x3zcnl2d6pcwswi5k77gm6ipnq8sjpnpgz3sf9h540lzlrhn";
    "aarch64-linux" = "sha256:0xx10pk32k0a62yyzcma6m8yqdlx5nmbb0084sz0yjhs8zagh931";
    "x86_64-linux" = "sha256:08m0k3l4kn7lng4vvx7janzan2qsnmmbvkfjqyz512fnh6rl9cn7";
  };
  xla_suffix = builtins.getAttr system xla_suffix_map;
  xla_ext = fetchTarball {
    url = "https://github.com/elixir-nx/xla/releases/download/v${xla_ext_ver}/xla_extension-${xla_ext_ver}-${system}-${xla_suffix}";
    sha256 = builtins.getAttr system xla_sha256_map;
  };
in
  xla_ext
