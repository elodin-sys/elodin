{
  base,
  lib,
  uv,
  python3,
  patch,
}: let
  patches = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [./apply.py ./bevy-cesium.patch ./tiles-core.patch ./elodin-editor.patch];
  };
in
  base.overrideAttrs (old: {
    pname = "elodin-web-progressive";
    nativeBuildInputs = old.nativeBuildInputs ++ [uv patch];
    postPatch =
      (old.postPatch or "")
      + ''
        patch --batch -p1 < ${patches}/elodin-editor.patch
        uv run --no-cache --offline --no-project --python ${python3}/bin/python \
          ${patches}/apply.py "$cargoDepsCopy"
      '';
  })
