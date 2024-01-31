{ config, self', pkgs, lib, ... }:
let
  cfg = config.buildkite-test-collector;
in
{
  options.buildkite-test-collector = {
    version = lib.mkOption { type = lib.types.str; };
    hash = lib.mkOption { type = lib.types.str; };
    cargoHash = lib.mkOption { type = lib.types.str; };
  };
  config = {
    packages.buildkite-test-collector = pkgs.rustPlatform.buildRustPackage {
      pname = "buildkite-test-collector";
      version = cfg.version;

      src = pkgs.fetchFromGitHub {
        owner = "buildkite";
        repo = "test-collector-rust";
        rev = cfg.version;
        hash = cfg.hash;
      };
      postConfigure = ''
        cargo metadata --offline
      '';
      cargoHash = cfg.cargoHash;
    };
  };
}
