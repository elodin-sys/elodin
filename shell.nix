{pkgs ? import <nixpkgs> {}}:
with pkgs;
  mkShell {
    buildInputs = [
      bazelisk
      python3
      clang_20
      # apple-sdk_15
      # (darwinMinVersionHook "15.0")
    ];
  }
