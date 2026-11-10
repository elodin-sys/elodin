import json
import sqlite3
import struct

import pytest

from scripts.export_photo3d_cache import ExportError, export_cache


ROOT = "https://tile.googleapis.com/v1/3dtiles/root.json?key=PRIVATE_KEY"
CHILD = "https://tile.googleapis.com/v1/3dtiles/child.json?session=PRIVATE_SESSION&key=PRIVATE_KEY"
TILE = "https://tile.googleapis.com/v1/3dtiles/tile.glb?key=ANOTHER_KEY&session=OLD_SESSION"


def glb(external=False):
    document = {"asset": {"version": "2.0", "copyright": "Google; provider"}}
    if external:
        document["buffers"] = [{"uri": "external.bin", "byteLength": 4}]
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    return struct.pack("<4sIIII", b"glTF", 2, 20 + len(payload), len(payload), 0x4E4F534A) + payload


def tileset(children):
    return json.dumps(
        {
            "asset": {"version": "1.1"},
            "geometricError": 100,
            "root": {
                "boundingVolume": {"sphere": [1, 2, 3, 10]},
                "geometricError": 50,
                "refine": "REPLACE",
                "children": [
                    {
                        "boundingVolume": {"sphere": [1, 2, 3, 2]},
                        "geometricError": 0,
                        "content": {"uri": uri},
                    }
                    for uri in children
                ],
            },
        }
    ).encode()


def cache(path, rows):
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE CacheItemTable(requestUrl TEXT, responseData BLOB, "
            "lastAccessedTime INTEGER, responseStatusCode INTEGER)"
        )
        db.executemany(
            "INSERT INTO CacheItemTable VALUES (?, ?, ?, ?)",
            [(url, data, i, 200) for i, (url, data) in enumerate(rows)],
        )
    return path


def test_real_bytes_and_nested_hierarchy_move_without_credentials(tmp_path):
    database = cache(
        tmp_path / "cache.sqlite",
        [
            (ROOT, tileset([CHILD, "missing.json?key=PRIVATE_KEY"])),
            (CHILD, tileset(["tile.glb?session=NEW_SESSION&key=NEW_KEY"])),
            (TILE, glb()),
        ],
    )
    before = database.read_bytes()
    output = tmp_path / "terrain"
    report = export_cache(database, output)
    assert database.read_bytes() == before
    assert report["tiles"] == 1
    assert report["tilesets"] == 2
    assert report["missing_content_references"] == 1
    assert report["coverage"] == "cached_subset"
    moved = tmp_path / "moved"
    output.rename(moved)
    root = json.loads((moved / "tileset.json").read_bytes())
    children = root["root"]["children"]
    assert len(children) == 1
    child_file = moved / children[0]["content"]["uri"]
    child = json.loads(child_file.read_bytes())
    tile_file = child_file.parent / child["root"]["children"][0]["content"]["uri"]
    assert tile_file.read_bytes() == glb()
    for file in moved.rglob("*"):
        if file.is_file():
            content = file.read_bytes()
            assert b"PRIVATE" not in content
            assert b"SESSION" not in content
            assert b"tile.googleapis.com" not in content


@pytest.mark.parametrize("bad", [b"glTF", glb(external=True)])
def test_incomplete_geometry_fails_without_publishing_a_package(tmp_path, bad):
    database = cache(tmp_path / "cache.sqlite", [(ROOT, tileset([TILE])), (TILE, bad)])
    output = tmp_path / "terrain"
    with pytest.raises(ExportError):
        export_cache(database, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".terrain-*"))


def test_cycle_is_reported_without_publishing(tmp_path):
    database = cache(
        tmp_path / "cache.sqlite", [(ROOT, tileset([CHILD])), (CHILD, tileset([ROOT]))]
    )
    with pytest.raises(ExportError, match="Cyclic"):
        export_cache(database, tmp_path / "terrain")
    assert not (tmp_path / "terrain").exists()


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "terrain"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_text("valuable")
    with pytest.raises(ExportError, match="already exists"):
        export_cache(tmp_path / "absent.sqlite", output)
    assert keep.read_text() == "valuable"


def test_missing_geometry_cannot_look_like_a_successful_export(tmp_path):
    database = cache(tmp_path / "cache.sqlite", [(ROOT, tileset([TILE]))])
    with pytest.raises(ExportError, match="no reachable"):
        export_cache(database, tmp_path / "terrain")


@pytest.mark.parametrize("uri", ["schema.json", "../outside.json", "https://example.com/schema"])
def test_unhandled_schema_dependency_cannot_escape_or_leave_an_incomplete_package(tmp_path, uri):
    root = json.loads(tileset([TILE]))
    root["schemaUri"] = uri
    database = cache(tmp_path / "cache.sqlite", [(ROOT, json.dumps(root).encode()), (TILE, glb())])
    with pytest.raises(ExportError, match="reference"):
        export_cache(database, tmp_path / "terrain")
    assert not (tmp_path / "terrain").exists()
