{ config, self', pkgs, lib, flakeInputs, rustToolchain, ... }:
let
  wasmTarget = "wasm32-unknown-unknown";
  rustWithWasmTarget = rustToolchain.override {
    targets = [wasmTarget];
  };
  craneLib = (flakeInputs.crane.mkLib pkgs).overrideToolchain rustWithWasmTarget;
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
    CARGO_BUILD_TARGET = wasmTarget;
    CARGO_PROFILE = "wasm-release";
  };
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  wasm = craneLib.buildPackage (commonArgs // {
    inherit cargoArtifacts;
  });
  bundle = pkgs.runCommand "editor-web-bundle" {} ''
    ${pkgs.wasm-bindgen-cli}/bin/wasm-bindgen --out-dir $out --target web ${wasm}/bin/editor-web.wasm
  '';
in
{
  packages.editor-web = bundle;
}
