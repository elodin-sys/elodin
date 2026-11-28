"""Add optional browser diagnostics to the pinned, writable Cargo dependency."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib


def main() -> None:
    patch = Path(__file__).resolve().with_name("tiles-core.patch")
    crates = [
        manifest.parent
        for manifest in Path(sys.argv[1]).resolve().rglob("Cargo.toml")
        if tomllib.loads(manifest.read_text()).get("package", {}).get("name") == "tiles-core"
    ]
    if not crates:
        raise SystemExit("Pinned tiles-core source not found")
    for crate in crates:
        subprocess.run(
            ["patch", "--batch", "--forward", "-p1", "-d", str(crate), "-i", str(patch)],
            check=True,
        )
        checksum = next(
            (
                p / ".cargo-checksum.json"
                for p in [crate, *crate.parents]
                if (p / ".cargo-checksum.json").exists()
            ),
            None,
        )
        if checksum is not None:
            value = json.loads(checksum.read_text())
            for name in value["files"]:
                value["files"][name] = hashlib.sha256(
                    (checksum.parent / name).read_bytes()
                ).hexdigest()
            checksum.write_text(json.dumps(value))


if __name__ == "__main__":
    main()
