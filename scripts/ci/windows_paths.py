#!/usr/bin/env python3
"""Windows-safe path helpers for CI baselines and CSV export filenames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath

WINDOWS_INVALID_CHARS = frozenset('<>:"|?*')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def windows_safe_name(name: str) -> str:
    return name.replace("_>_", "_to_").replace(">", "to")


def windows_safe_rel_path(rel_path: str) -> str:
    return windows_safe_name(rel_path)


def windows_path_error(path: str) -> str | None:
    for part in PurePosixPath(path).parts:
        if any(char in WINDOWS_INVALID_CHARS for char in part):
            return f"contains a Windows-invalid character: {path}"

        stem = part.rstrip(" .").split(".", 1)[0].upper()
        if stem in WINDOWS_RESERVED_NAMES:
            return f"uses a Windows-reserved path component: {path}"

        if part.endswith((" ", ".")):
            return f"ends a path component with a space or dot: {path}"

    return None


def sanitize_export_filenames(root: Path) -> list[tuple[Path, Path]]:
    """Rename files under *root* so names are Windows-checkout-safe."""
    renamed: list[tuple[Path, Path]] = []
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.is_file():
            continue
        new_name = windows_safe_name(path.name)
        if new_name == path.name:
            continue
        dest = path.with_name(new_name)
        if dest.exists():
            raise FileExistsError(f"cannot sanitize {path.name}: {dest.name} already exists")
        path.rename(dest)
        renamed.append((path, dest))
    return renamed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sanitize = sub.add_parser(
        "sanitize-dir",
        help="Rename files in a directory, mapping '>' to 'to'",
    )
    sanitize.add_argument("root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "sanitize-dir":
        if not args.root.is_dir():
            print(f"FAIL: not a directory: {args.root}", file=sys.stderr)
            return 1
        for _old, new in sanitize_export_filenames(args.root):
            print(f"  renamed -> {new.name}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
