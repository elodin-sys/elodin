{
  pkgs,
  lib,
  crane,
  rustToolchain,
  system,
  ...
}: let
  xla_ext = pkgs.callPackage ./xla-ext.nix {};
  craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
  pname = (craneLib.crateNameFromCargoToml {cargoToml = ../../libs/nox-py/Cargo.toml;}).pname;
  version = (craneLib.crateNameFromCargoToml {cargoToml = ../../Cargo.toml;}).version;

  pyFilter = path: _type: builtins.match ".*py$" path != null;
  mdFilter = path: _type: builtins.match ".*nox-py.*md$" path != null;
  protoFilter = path: _type: builtins.match ".*proto$" path != null;
  assetFilter = path: _type: builtins.match ".*assets.*$" path != null;
  cppFilter = path: _type: builtins.match ".*[h|(cpp)|(cpp.jinja)]$" path != null;

  srcFilter = path: type:
    (pyFilter path type)
    || (mdFilter path type)
    || (protoFilter path type)
    || (assetFilter path type)
    || (cppFilter path type)
    || (craneLib.filterCargoSources path type);

  src = lib.cleanSourceWith {
    src = craneLib.path ./../..;
    filter = srcFilter;
  };

  arch = with pkgs;
    if stdenv.isDarwin
    then stdenv.hostPlatform.ubootArch
    else builtins.elemAt (lib.strings.splitString "-" system) 0;

  # Patch RPATHs after install so Python/Rust can find xla_extension without env vars.
  fixup = with pkgs;
    (lib.optionalString stdenv.isLinux ''
      # --- Linux: patch ELF rpaths for both libs and bins ---
      set -eu
      # Collect targets: *.so and versioned *.so.*, plus executables in $out/bin
      mapfile -d "" TARGETS < <(
        { find "$out" -type f \( -name "*.so" -o -name "*.so.*" \) -print0 2>/dev/null || true; }
        { if [ -d "$out/bin" ]; then find "$out/bin" -type f -perm -0100 -print0; fi; }
      )
      for f in "''${TARGETS[@]}"; do
        # Skip non-ELF files just in case
        if ! file -h "$f" | grep -q ELF; then continue; fi
        old="$(patchelf --print-rpath "$f" 2>/dev/null || true)"
        # Preserve old rpath and add our locations
        new="$old"
        [ -n "$new" ] && new="$new:"
        new+="${xla_ext}/lib:\$ORIGIN:\$ORIGIN/..:$out/lib:${stdenv.cc.cc.lib}/lib:${pkgs.gfortran.cc.lib}/lib"
        patchelf --set-rpath "$new" "$f" || true
      done
    '')
    + (lib.optionalString stdenv.isDarwin ''
      # --- macOS: add @rpath entries so the loader can find xla_extension ---
      set -eu
      add_rpath_if_missing() {
        local file="$1" rp="$2"
        if ! otool -l "$file" | grep -A2 LC_RPATH | grep -q "path $rp (offset"; then
          install_name_tool -add_rpath "$rp" "$file" || true
        fi
      }

      # Patch both .so (Python extensions/bundles) and .dylib, plus any bins in $out/bin
      while IFS= read -r -d "" f; do
        # Only touch Mach-O files
        if ! file -h "$f" | grep -Eq 'Mach-O|bundle'; then continue; fi

        # Ensure typical relative rpaths work when colocating libs
        add_rpath_if_missing "$f" "@loader_path"
        add_rpath_if_missing "$f" "@loader_path/.."

        # Ensure absolute path to the unpacked xla_ext
        add_rpath_if_missing "$f" "${xla_ext}/lib"

        # If the binary/lib directly references an xla_extension by a non-store path,
        # rewrite it to our absolute one.
        for dep in $(otool -L "$f" | awk '{print $1}' | grep -E 'xla_extension|libxla'); do
          base="$(basename "$dep")"
          if [ -f "${xla_ext}/lib/$base" ]; then
            install_name_tool -change "$dep" "${xla_ext}/lib/$base" "$f" || true
          fi
        done
      done < <(
        { find "$out" -type f \( -name "*.so" -o -name "*.dylib" \) -print0 2>/dev/null || true; }
        { if [ -d "$out/bin" ]; then find "$out/bin" -type f -perm -0100 -print0; fi; }
      )
    '');

  commonArgs = {
    inherit pname version;
    inherit src;

    nativeBuildInputs = with pkgs;
      [maturin]
      ++ lib.optionals stdenv.isLinux [
        autoPatchelfHook
        patchelf
      ]
      ++ lib.optionals stdenv.isDarwin [
        fixDarwinDylibNames
        darwin.cctools
      ];
    buildInputs = with pkgs;
      [
        pkg-config
        python3
        openssl
        cmake
        gfortran
        gfortran.cc.lib
        xla_ext
      ]
      ++ lib.optionals stdenv.isDarwin [pkgs.libiconv];

    postFixup = fixup;

    cargoExtraArgs = "--package=nox-py";

    XLA_EXTENSION_DIR = "${xla_ext}";
    OPENSSL_DIR = "${pkgs.openssl.dev}";
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
    OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include/";
  };

  cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  clippy = craneLib.cargoClippy (
    commonArgs
    // {
      inherit cargoArtifacts;
      cargoClippyExtraArgs = "--all-targets -- --deny warnings";
    }
  );

  wheelName = "elodin";
  wheelPlatform =
    if pkgs.stdenv.isDarwin
    then "macosx_11_0"
    else "linux";
  wheelSuffix = "cp310-abi3-${wheelPlatform}_${arch}";
  wheel = craneLib.buildPackage (
    commonArgs
    // {
      inherit cargoArtifacts;
      doCheck = false;
      pname = "elodin";
      buildPhase = "maturin build --offline --target-dir ./target -m libs/nox-py/Cargo.toml --release";
      installPhase = "install -D target/wheels/${wheelName}-${version}-${wheelSuffix}.whl -t $out/";
    }
  );

  elodin = ps:
    ps.buildPythonPackage {
      pname = wheelName;
      format = "wheel";
      version = version;
      src = "${wheel}/${wheelName}-${version}-${wheelSuffix}.whl";
      doCheck = false;
      propagatedBuildInputs = with ps; [
        jax
        jaxlib
        typing-extensions
        numpy
        polars
        pytest
        xla_ext
        pkgs.gfortran.cc.lib
      ];
      buildInputs = [xla_ext pkgs.gfortran.cc.lib];
      nativeBuildInputs = with pkgs; (
        lib.optionals stdenv.isLinux [autoPatchelfHook patchelf]
        ++ lib.optionals stdenv.isDarwin [fixDarwinDylibNames darwin.cctools]
      );
      postFixup = fixup;
      pythonImportsCheck = [wheelName];
    };
  py = elodin pkgs.python3Packages;
in {
  inherit py clippy;
}
