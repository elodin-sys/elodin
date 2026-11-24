"""Apply the experiment to Cargo's writable, pinned dependency sources."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib


def apply(crate: Path, patch: Path) -> None:
    subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-d", str(crate), "-i", str(patch)],
        check=True,
    )
    checksum = next(
        (
            parent / ".cargo-checksum.json"
            for parent in [crate, *crate.parents]
            if (parent / ".cargo-checksum.json").exists()
        ),
        None,
    )
    if checksum is not None:
        value = json.loads(checksum.read_text())
        for name in value["files"]:
            value["files"][name] = hashlib.sha256((checksum.parent / name).read_bytes()).hexdigest()
        checksum.write_text(json.dumps(value))


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    patches = Path(__file__).resolve().parent
    found = []
    for manifest in root.rglob("Cargo.toml"):
        package = tomllib.loads(manifest.read_text()).get("package", {}).get("name")
        if package in ("bevy-cesium", "tiles-core"):
            found.append((package, manifest.parent))
    if {package for package, _ in found} != {"bevy-cesium", "tiles-core"}:
        raise SystemExit(f"Expected both pinned terrain crates, found {found}")
    # Cargo vendors both the standalone core crate and its copy in the Git source.
    for package, crate in found:
        apply(crate, patches / f"{package}.patch")


if __name__ == "__main__":
    main()
