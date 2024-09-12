{ config, self', pkgs, lib, flakeInputs, rustToolchain, ... }:
let
  craneLib = (flakeInputs.crane.mkLib pkgs).overrideToolchain rustToolchain;
  crateName = craneLib.crateNameFromCargoToml { cargoToml = ../apps/editor-web/Cargo.toml; };
  src = pkgs.nix-gitignore.gitignoreSource [] ../.;
  commonArgs = {
    inherit (crateName) pname version;
    inherit src;
    doCheck = false;
    cargoExtraArgs = "--package=${crateName.pname}";
    buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
      pkgs.libiconv
    ];
    CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
    CARGO_PROFILE = "wasm-release";
    RUSTFLAGS = "--cfg=web_sys_unstable_apis";
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  wasm = craneLib.buildPackage (commonArgs // {
    inherit cargoArtifacts;
  });
  wasm-bindgen-cli = config.packages.wasm-bindgen-cli;
  bundle = pkgs.runCommand "editor-web-bundle" {} ''
    ${wasm-bindgen-cli}/bin/wasm-bindgen --out-dir $out --target web ${wasm}/bin/editor-web.wasm
  '';
in
{
  packages.editor-web = bundle;
}
