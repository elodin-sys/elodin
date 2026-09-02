"""Watch a Python schematic script and push on change (Phase 3).

Script contract: define ``build() -> elodin.ui.Schematic``.

Example::

    python -m elodin.ui.watch examples/db-client/schematic.py --db 127.0.0.1:2240
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"elodin_ui_watch_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_schematic(path: Path):
    mod = _load_module(path)
    if not hasattr(mod, "build"):
        raise RuntimeError(f"{path} must define build() -> Schematic")
    schematic = mod.build()
    emit = getattr(schematic, "emit_kdl", None)
    if not callable(emit):
        raise TypeError("build() must return an elodin.ui.Schematic")
    return schematic


def _set_build_error(db: str, message: str | None) -> None:
    import elodin.ui as ui

    ui.set_build_error(db, message)


def _assets_http_url(db: str, key: str) -> str:
    host, _, port = db.rpartition(":")
    host = host.strip("[]")
    if host in ("", "0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{int(port) + 1}/{key}"


def _fetch_overlay_kdl(db: str, schematic_key: str) -> str | None:
    import elodin.ui as ui

    url = _assets_http_url(db, ui.overlay_key(schematic_key))
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            if resp.status != 200:
                return None
            text = resp.read().decode()
        return text or None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def run_once(path: Path, db: str, *, quiet: bool = False) -> bool:
    import elodin.ui as ui

    try:
        schematic = _build_schematic(path)
        if overlay := _fetch_overlay_kdl(db, "schematics/main.kdl"):
            schematic = ui.apply_overlay(schematic, overlay)
        ui.push(schematic, db)
        _set_build_error(db, None)
        if not quiet:
            print(f"[ui watch] pushed {path.name} → {db}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        detail = traceback.format_exc()
        print(f"[ui watch] build failed:\n{detail}", file=sys.stderr, flush=True)
        try:
            _set_build_error(db, err)
        except Exception as push_exc:  # noqa: BLE001
            print(f"[ui watch] could not publish build error: {push_exc}", file=sys.stderr)
        return False


def watch(path: Path, db: str, *, debounce_ms: int = 200) -> int:
    try:
        from watchfiles import watch as watch_files
    except ImportError as exc:
        raise SystemExit(
            "watchfiles is required for `elodin ui watch` — "
            "re-run `just install py` or `pip install watchfiles`"
        ) from exc

    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"script not found: {path}")

    print(f"[ui watch] watching {path} → {db}", flush=True)
    last_good = run_once(path, db)

    for changes in watch_files(path, debounce=debounce_ms):
        # Any change to the script triggers a rebuild.
        _ = changes
        ok = run_once(path, db)
        if ok:
            last_good = True
        elif last_good:
            print("[ui watch] keeping last-good schematic in the DB", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="elodin ui watch")
    parser.add_argument("script", type=Path, help="Python schematic script with build()")
    parser.add_argument(
        "--db",
        default="127.0.0.1:2240",
        help="Impeller DB address (default: 127.0.0.1:2240)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Build and push once, then exit",
    )
    parser.add_argument(
        "--debounce-ms",
        type=int,
        default=200,
        help="Debounce interval for file changes",
    )
    args = parser.parse_args(argv)

    if args.once:
        return 0 if run_once(args.script, args.db) else 1

    try:
        return watch(args.script, args.db, debounce_ms=args.debounce_ms)
    except KeyboardInterrupt:
        print("\n[ui watch] stopped", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
