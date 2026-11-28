"""Regenerate the diagnostics-only patch from an edited dependency checkout."""

import difflib
from pathlib import Path
import sys

base, edited = map(Path, sys.argv[1:3])
name = Path("crates/tiles-core/src/lib.rs")
patch = difflib.unified_diff(
    (base / name).read_text().splitlines(keepends=True),
    (edited / name).read_text().splitlines(keepends=True),
    fromfile="a/src/lib.rs",
    tofile="b/src/lib.rs",
)
Path("experiments/terrain-progressive/tiles-core.patch").write_text(
    "".join(patch).replace("\n \n", "\n\n")
)
