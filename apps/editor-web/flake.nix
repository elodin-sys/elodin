{
  inputs = {
    paracosm.url = "path:../../.";
    nixpkgs.follows = "paracosm/nixpkgs";
    rust-overlay.follows = "paracosm/rust-overlay";
    flake-utils.follows = "paracosm/flake-utils";
    crane.url = "github:ipetkov/crane";
  };

  outputs = inputs:
    with inputs;
      flake-utils.lib.eachDefaultSystem (
        system: let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [rust-overlay.overlays.default];
          };
          wasmTarget = "wasm32-unknown-unknown";
          rustWithWasmTarget = pkgs.rust-bin.stable."1.73.0".default.override {
            targets = [wasmTarget];
          };
          craneLib = crane.mkLib pkgs;
          craneLibWasm = craneLib.overrideToolchain rustWithWasmTarget;
          args = {
            pname = "editor-web";
            version = "0.0.1";
            src = pkgs.nix-gitignore.gitignoreSource [] ../..;
            doCheck = false;
            cargoExtraArgs = "--package=editor-web";
            CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
          };
          cargoArtifacts =
            craneLibWasm.buildDepsOnly args
            // {
              buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
                pkgs.libiconv
              ];
            };
          wasm =
            craneLibWasm.buildPackage args
            // {
              inherit cargoArtifacts;
            };
          bundle = pkgs.runCommand "editor-web-bundle" {} ''
            ${pkgs.wasm-bindgen-cli}/bin/wasm-bindgen --out-dir $out --target web ${wasm}/bin/editor-web.wasm
          '';
        in {
          packages = {
            default = bundle;
          };
        }
      );
}
