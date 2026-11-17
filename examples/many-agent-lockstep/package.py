#!/usr/bin/env python3
"""Build a sanitized, relative-path handoff archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import tarfile
import tempfile
from pathlib import Path


EXAMPLE = Path(__file__).resolve().parent
TEXT_SUFFIXES = {".json", ".log", ".md", ".txt", ".csv", ".py", ".rs", ".toml"}
EXCLUDED_PARTS = {"build", "__pycache__", "db", "tar" + "get"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_output", type=Path, nargs="+")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profile_output/many-agent-lockstep-handoff.tar.gz"),
    )
    parser.add_argument("--deny-file", type=Path)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sanitize_text(text: str) -> str:
    replacements = {
        str(Path.home()): "$HOME",
        str(EXAMPLE.parents[1]): ".",
        socket.gethostname(): "<host>",
    }
    for old, new in replacements.items():
        if old:
            text = text.replace(old, new)
    text = re.sub(r"\$HOME/[^\s\"\]]+", "<path>", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<address>", text)
    return text


def copy_tree(source: Path, destination: Path, *, sanitize: bool = True) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.is_dir() or path.suffix == ".pyc":
            continue
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            output.write_text(sanitize_text(text) if sanitize else text, encoding="utf-8")
        else:
            shutil.copyfile(path, output)


def denied_patterns(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def normalize_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def main() -> int:
    args = parse_args()
    benchmark_outputs = [path.resolve() for path in args.benchmark_output]
    if any(not (path / "matrix.json").is_file() for path in benchmark_outputs):
        raise RuntimeError("benchmark output is missing matrix.json")
    patterns = denied_patterns(args.deny_file)
    with tempfile.TemporaryDirectory(prefix="many-agent-handoff-") as temp:
        root = Path(temp) / "many-agent-lockstep"
        source_dir = root / "example"
        copy_tree(EXAMPLE, source_dir, sanitize=False)
        for benchmark_output in benchmark_outputs:
            copy_tree(
                benchmark_output,
                root / "evidence" / benchmark_output.name,
            )
        facts = {
            "schema": "elodin.many_agent.handoff.v1",
            "architecture": platform.machine(),
            "os": platform.system(),
            "cpu_count": os.cpu_count(),
        }
        (root / "vm.json").write_text(json.dumps(facts, indent=2) + "\n")
        for pattern in patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in TEXT_SUFFIXES:
                    if regex.search(path.read_text(encoding="utf-8", errors="replace")):
                        raise RuntimeError(f"denied content in {path.relative_to(root)}")
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                files.append(
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "sha256": digest(path),
                    }
                )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "elodin.many_agent.manifest.v1",
                    "files": files,
                },
                indent=2,
            )
            + "\n"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w:gz") as archive:
            archive.add(root, arcname=root.name, filter=normalize_tar_info)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
