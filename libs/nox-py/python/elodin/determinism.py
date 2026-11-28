"""Compare deterministic witnesses, and replay `pre_commit` hooks from them.

`main` is the `elodin determinism` command. It exits 0 when the witnesses match or the replay
reproduces every byte, 1 at a mismatch, and 2 on an error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .elodin import _witness_compare, _witness_inspect, _witness_replay_hook


def compare(left: str | Path, right: str | Path, stop_at_first: bool = False) -> dict:
    """Compares two witness files tick by tick. The result has `identical`, `first_mismatch`
    (or `None`), the counts of ticks compared and mismatched, and each side's summary. Raises
    `ValueError` when the files are unreadable or their schemas differ."""
    return json.loads(_witness_compare(str(left), str(right), stop_at_first))


def replay_hook(trace: str | Path, library: str | Path, symbol: str, live: bool = False) -> dict:
    """Replays the native hook `symbol` of the shared library `library` on the rows `trace` kept
    exactly, and compares what it writes with the recording. With `live`, the hook does its own
    I/O instead of reading the recorded external channels."""
    return json.loads(_witness_replay_hook(str(trace), str(library), symbol, live))


def inspect(trace: str | Path) -> dict:
    """Validate a witness and return its header, trailer, and tick inventory."""
    return json.loads(_witness_inspect(str(trace)))


def describe(mismatch: dict) -> str:
    """A mismatch as the lines `elodin determinism` prints."""
    lines = ["first mismatch", f"  tick: {mismatch['tick']}"]
    for key in ("stage", "hook"):
        if mismatch.get(key) is not None:
            lines.append(f"  {key}: {mismatch[key]}")
    if mismatch.get("kind") is not None:
        lines.append(f"  {mismatch['kind']}: {mismatch['record']}")
    for key in ("entity", "row", "byte"):
        if mismatch.get(key) is not None:
            lines.append(f"  {key}: {mismatch[key]}")
    for key in ("left", "right"):
        if mismatch.get(key) is not None:
            lines.append(f"  {key}: 0x{mismatch[key]:02x}")
    if mismatch.get("note"):
        lines.append(f"  note: {mismatch['note']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="elodin determinism", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare_parser = commands.add_parser("compare", help="report where two witnesses first differ")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--stop-at-first", action="store_true")
    replay_parser = commands.add_parser("replay-hook", help="replay a native hook from a witness")
    replay_parser.add_argument("trace", type=Path)
    replay_parser.add_argument("--library", type=Path, required=True)
    replay_parser.add_argument("--symbol", required=True)
    replay_parser.add_argument("--live", action="store_true")
    inspect_parser = commands.add_parser(
        "inspect", help="print a validated witness header, schema and tick inventory"
    )
    inspect_parser.add_argument("trace", type=Path)
    inspect_parser.add_argument("--json", action="store_true")
    inspect_parser.add_argument("--schema", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            result = compare(args.left, args.right, args.stop_at_first)
        elif args.command == "inspect":
            result = inspect(args.trace)
        else:
            result = replay_hook(args.trace, args.library, args.symbol, args.live)
    except (ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.command == "inspect":
        if args.schema:
            args.schema.write_text(json.dumps(result["header"]["schema"], indent=2) + "\n")
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            trailer = result.get("trailer") or {}
            print(
                f"{result['ticks']} ticks ({result['raw_ticks']} raw), "
                f"root {trailer.get('root', 'incomplete')}"
            )
        return 0
    mismatch = result["first_mismatch"]
    if args.command == "compare":
        if mismatch is None:
            print(f"identical: {result['ticks']} ticks, root {result['left']['root']}")
            return 0
        print(describe(mismatch))
        if not args.stop_at_first:
            print(f"{result['mismatched_ticks']} of {result['ticks']} ticks differ")
        return 1
    if mismatch is None:
        print(
            f"reproduced: {result['ticks']} ticks, {result['records']} records, "
            f"{result['bytes']} bytes; {result['skipped_ticks']} ticks had digests only"
        )
        return 0
    print(describe(mismatch))
    return 1
