{
  lib,
  # LFS-tracked editor assets; a `git+file?rev=` export holds pointers, so packages
  # that never embed them must omit them to build identically from git history.
  editorAssets ? true,
}: let
  root = toString ../..;
  workspaceDirs = ["/apps" "/libs" "/fsw" "/examples"];
  rootFiles = ["/Cargo.toml" "/Cargo.lock" "/rust-toolchain.toml" "/rustfmt.toml"];
  # Non-Rust entries are read at compile time: tonic protos, include_str!/embedded_asset!
  # files, and the Python sources and readme packed into the maturin wheel.
  suffixes =
    [
      "Cargo.toml"
      ".rs"
      ".c"
      ".h"
      ".cpp"
      ".hpp"
      ".jinja"
      ".proto"
      ".md"
      ".json"
      "pyproject.toml"
      ".py"
      ".pyi"
      "py.typed"
      ".wgsl"
      ".ttf"
      ".png"
      ".bin"
      "logo.txt"
    ]
    ++ lib.optionals editorAssets [".ktx2" ".glb"];
  # /examples contributes only its Rust controller crates, not the Python examples.
  exampleSuffixes = ["Cargo.toml" ".rs"];
in
  lib.cleanSourceWith {
    src = ../..;
    filter = path: type: let
      relPath = lib.removePrefix root path;
      under = dir: relPath == dir || lib.hasPrefix "${dir}/" relPath;
    in
      # Submodule contents differ between a checkout and a `git+file?rev=` export.
      if under "/examples/betaflight-sitl/betaflight"
      then false
      else if !(lib.any under workspaceDirs)
      then lib.elem relPath rootFiles
      else
        type
        == "directory"
        || lib.any (suffix: lib.hasSuffix suffix relPath) (
          if under "/examples"
          then exampleSuffixes
          else suffixes
        );
  }
