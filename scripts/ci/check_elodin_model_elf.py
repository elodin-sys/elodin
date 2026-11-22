#!/usr/bin/env python3
"""Reject a model library that cannot be delivered as native RHEL 8 x86-64."""

import argparse
import re
import subprocess
from pathlib import Path


def readelf(path: Path, *args: str) -> str:
    return subprocess.run(
        ["readelf", *args, path],
        universal_newlines=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout


parser = argparse.ArgumentParser()
parser.add_argument("library", type=Path)
parser.add_argument("--max-glibc", default="2.28")
args = parser.parse_args()

header = readelf(args.library, "-h")
if "Machine:" not in header or "X86-64" not in header:
    raise SystemExit(f"{args.library} is not an x86-64 ELF library")

dynamic = readelf(args.library, "-d")
for line in dynamic.splitlines():
    if ("RPATH" in line or "RUNPATH" in line) and "/nix/store" in line:
        raise SystemExit(f"{args.library} contains a Nix store runpath: {line.strip()}")

versions = readelf(args.library, "--version-info")
required = {
    tuple(map(int, match.groups())) for match in re.finditer(r"Name: GLIBC_(\d+)\.(\d+)", versions)
}
maximum = tuple(map(int, args.max_glibc.split(".")))
newer = sorted(version for version in required if version > maximum)
if newer:
    text = ", ".join(".".join(map(str, version)) for version in newer)
    raise SystemExit(f"{args.library} requires GLIBC newer than {args.max_glibc}: {text}")

print(
    f"{args.library}: x86-64, no Nix runpath, "
    f"maximum GLIBC_{'.'.join(map(str, max(required, default=(0, 0))))}"
)
