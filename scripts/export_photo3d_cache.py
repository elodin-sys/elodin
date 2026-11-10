#!/usr/bin/env python3
"""Export the Google Photo 3D subset already held in Cesium Native's cache.

This prototype makes no network requests. Missing branches are pruned and
reported, so the result is a snapshot of cached coverage, not a complete region.
Run from the repository root with `uv run python scripts/export_photo3d_cache.py`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import posixpath
import shutil
import sqlite3
import struct
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlsplit, urlunsplit


class ExportError(ValueError):
    """Cache data cannot be exported as a self-contained 3D Tiles package."""


def cache_key(url: str) -> str:
    parts = urlsplit(url)
    # Google sessions and credentials identify access, not the tile resource.
    query = sorted((k, v) for k, v in parse_qsl(parts.query) if k not in {"key", "session"})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def validate_glb(data: bytes) -> None:
    if len(data) < 20:
        raise ExportError("Truncated GLB in the cache")
    magic, version, size, json_size, chunk_type = struct.unpack_from("<4sIIII", data)
    if magic != b"glTF" or version != 2 or size != len(data) or chunk_type != 0x4E4F534A:
        raise ExportError("Invalid GLB header in the cache")
    if json_size > len(data) - 20:
        raise ExportError("Truncated GLB JSON chunk")
    document = json.loads(data[20 : 20 + json_size])
    for group in ("buffers", "images"):
        for item in document.get(group, []):
            uri = item.get("uri")
            if uri is not None and not uri.startswith("data:"):
                raise ExportError("GLB has an external dependency; export is incomplete")


class Exporter:
    def __init__(self, db: sqlite3.Connection, directory: Path):
        self.db = db
        self.directory = directory
        self.rows: dict[str, int] = {}
        for rowid, url in db.execute(
            "SELECT rowid, requestUrl FROM CacheItemTable "
            "WHERE responseStatusCode=200 ORDER BY lastAccessedTime DESC, rowid DESC"
        ):
            if urlsplit(url).hostname == "tile.googleapis.com":
                self.rows.setdefault(cache_key(url), rowid)
        self.root = cache_key("https://tile.googleapis.com/v1/3dtiles/root.json")
        self.exported: dict[str, str | None] = {}
        self.visiting: set[str] = set()
        self.missing: set[str] = set()
        self.tiles = 0
        self.tilesets = 0
        self.byte_count = 0

    def write(self, filename: str, data: bytes) -> None:
        destination = self.directory / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        self.byte_count += len(data)

    def resource(self, url: str) -> str | None:
        key = cache_key(url)
        if key in self.visiting:
            raise ExportError("Cyclic tileset references in the cache")
        if key in self.exported:
            return self.exported[key]
        row = self.rows.get(key)
        if row is None:
            self.missing.add(key)
            return None
        (data,) = self.db.execute(
            "SELECT responseData FROM CacheItemTable WHERE rowid=?", (row,)
        ).fetchone()
        if not data:
            raise ExportError("Empty cached response")
        name = hashlib.sha256(key.encode()).hexdigest()
        self.visiting.add(key)
        try:
            if data.startswith(b"glTF"):
                validate_glb(data)
                filename = f"tiles/{name}.glb"
                self.write(filename, data)
                self.tiles += 1
            else:
                document = json.loads(data)
                if not isinstance(document, dict) or "root" not in document:
                    raise ExportError("Cached JSON is not a 3D Tiles tileset")
                filename = "tileset.json" if key == self.root else f"tiles/{name}.json"
                root = self.node(document["root"], key, filename)
                if root is None:
                    self.exported[key] = None
                    return None
                document["root"] = root
                self.check_references(document, self.directory / filename)
                payload = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
                self.write(filename, payload)
                self.tilesets += 1
            self.exported[key] = filename
            return filename
        finally:
            self.visiting.remove(key)

    def node(self, original: dict, base: str, filename: str) -> dict | None:
        if "implicitTiling" in original:
            raise ExportError("Implicit tilesets are not supported by this prototype")
        node = copy.deepcopy(original)
        contents = []
        for content in [node["content"]] if "content" in node else node.get("contents", []):
            uri = content.get("uri", content.get("url"))
            if not isinstance(uri, str):
                raise ExportError("Tile content has no URI")
            target = self.resource(urljoin(base, uri))
            if target is not None:
                content.pop("url", None)
                content["uri"] = posixpath.relpath(target, posixpath.dirname(filename) or ".")
                contents.append(content)
        node.pop("content", None)
        node.pop("contents", None)
        if len(contents) == 1:
            node["content"] = contents[0]
        elif contents:
            node["contents"] = contents
        children = []
        for child in node.pop("children", []):
            exported = self.node(child, base, filename)
            if exported is not None:
                children.append(exported)
        if children:
            node["children"] = children
        return node if contents or children else None

    def check_references(self, value: object, source: Path) -> None:
        """Reject resource references we did not localize, including extensions."""
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {"uri", "url", "schemauri"} and isinstance(item, str):
                    parts = urlsplit(item)
                    if parts.scheme or parts.netloc or parts.query or parts.fragment:
                        raise ExportError("An external or authenticated resource reference remains")
                    target = (source.parent / unquote(parts.path)).resolve()
                    if not target.is_relative_to(self.directory.resolve()) or not target.is_file():
                        raise ExportError(
                            "A resource reference is outside or missing from the package"
                        )
                self.check_references(item, source)
        elif isinstance(value, list):
            for item in value:
                self.check_references(item, source)


def export_cache(cache: Path, output: Path) -> dict:
    if output.exists():
        raise ExportError("Output already exists; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    reserved = False
    try:
        with closing(sqlite3.connect(f"{cache.resolve().as_uri()}?mode=ro", uri=True)) as db:
            db.execute("BEGIN")  # One consistent snapshot; never mutate the live cache.
            exporter = Exporter(db, stage)
            if exporter.root not in exporter.rows:
                raise ExportError("Google root tileset is not present in the cache")
            if exporter.resource(exporter.root) is None or exporter.tiles == 0:
                raise ExportError("The cache contains no reachable Google geometry")
            report = {
                "version": 1,
                "source": "Google Photorealistic 3D Tiles via Cesium Native cache",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "entrypoint": "tileset.json",
                "coverage": "cached_subset",
                "tiles": exporter.tiles,
                "tilesets": exporter.tilesets,
                "bytes": exporter.byte_count,
                "missing_content_references": len(exporter.missing),
                "note": "Only previously cached coverage is included. Other views may contain holes.",
            }
            (stage / "cache-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        output.mkdir()  # Reserve exclusively; do not overwrite an existing export.
        reserved = True
        stage.rename(output)
        return report
    except (sqlite3.Error, json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise ExportError("Unsupported or malformed Cesium cache data") from error
    finally:
        if stage.exists():
            shutil.rmtree(stage)
            if reserved:
                output.rmdir()


def main() -> None:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New portable terrain directory")
    parser.add_argument(
        "--cache", type=Path, default=cache_root / "bevy-cesium/terrain/requests.sqlite"
    )
    args = parser.parse_args()
    try:
        report = export_cache(args.cache, args.output)
    except (ExportError, OSError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
