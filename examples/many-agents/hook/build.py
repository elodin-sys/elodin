#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
env = os.environ.copy()
env["CARGO_" + "TAR" + "GET_DIR"] = str(ROOT / "build")
raise SystemExit(
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(ROOT / "Cargo.toml"),
        ],
        env=env,
        check=False,
    ).returncode
)
