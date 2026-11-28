{
  base,
  lib,
  uv,
  python3,
  patch,
}: let
  diagnostics = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [./apply.py ./tiles-core.patch];
  };
in
  base.overrideAttrs (old: {
    pname = "elodin-web-diagnostics";
    nativeBuildInputs = old.nativeBuildInputs ++ [uv patch];
    postPatch =
      (old.postPatch or "")
      + ''
        uv run --no-cache --offline --no-project --python ${python3}/bin/python \
          ${diagnostics}/apply.py "$cargoDepsCopy"
      '';
  })
