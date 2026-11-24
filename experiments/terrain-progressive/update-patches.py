"""Regenerate patches after editing the ignored terrain dependency copy."""

import difflib
from pathlib import Path
import sys


base = Path(sys.argv[1])
edited = Path(".local/terrain-progressive/bevy-cesium")
output = Path("experiments/terrain-progressive")
groups = {
    "tiles-core": ["crates/tiles-core/src/lib.rs", "crates/tiles-core/src/tests.rs"],
    "bevy-cesium": [
        "src/render/contract.rs",
        "src/browser.rs",
        "src/backend.rs",
        "src/render/mod.rs",
        "src/render/readiness.rs",
    ],
}
for package, files in groups.items():
    chunks = []
    for name in files:
        relative = name.removeprefix("crates/tiles-core/") if package == "tiles-core" else name
        chunks.extend(
            difflib.unified_diff(
                (base / name).read_text().splitlines(keepends=True),
                (edited / name).read_text().splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    (output / f"{package}.patch").write_text("".join(chunks))
