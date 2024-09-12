{ config, self', pkgs, lib, ... }:
let
  cfg = config.wasm-bindgen-cli;
in
{
  options.wasm-bindgen-cli = {
    version = lib.mkOption { type = lib.types.str; };
    hash = lib.mkOption { type = lib.types.str; };
    cargoHash = lib.mkOption { type = lib.types.str; };
  };
  config = {
    packages.wasm-bindgen-cli = pkgs.rustPlatform.buildRustPackage rec {
      pname = "wasm-bindgen-cli";
      version = cfg.version;
      cargoHash = cfg.cargoHash;

      src = pkgs.fetchCrate {
        inherit pname version;
        hash = cfg.hash;
      };
    };
  };
}
