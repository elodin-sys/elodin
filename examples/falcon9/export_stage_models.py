"""Split the reference GLB without changing mesh geometry or model origins.

Run from the repository root with ``uv run examples/falcon9/export_stage_models.py``.
"""

import copy
import json
import struct
from pathlib import Path


def split_model(source: bytes, *, booster: bool) -> bytes:
    magic, version, length = struct.unpack_from("<4sII", source)
    assert (magic, version, length) == (b"glTF", 2, len(source))
    json_length, kind = struct.unpack_from("<I4s", source, 12)
    assert kind == b"JSON"
    model = json.loads(source[20 : 20 + json_length])
    binary_offset = 20 + json_length
    binary_length, kind = struct.unpack_from("<I4s", source, binary_offset)
    assert kind == b"BIN\0"
    binary = source[binary_offset + 8 : binary_offset + 8 + binary_length]
    assert not model.get("animations") and not any("skin" in node for node in model["nodes"])
    # The reference includes an unused skin; no mesh is skinned.
    model.pop("skins", None)
    assert not model.get("extensionsRequired")

    excluded = (
        {"stage_two_f9-v1.2_rig", "m1d-vac nozzle", "left fairing shell", "right fairing shell"}
        if booster
        else {"stage_one_f9-v1.2_rig", "stage one rcs"}
    )
    excluded.add("Sun")
    nodes = model["nodes"]
    assert excluded <= {node.get("name") for node in nodes}
    kept = []

    def visit(index):
        if nodes[index].get("name") in excluded:
            return
        kept.append(index)
        for child in nodes[index].get("children", []):
            visit(child)

    scene = model["scenes"][model.get("scene", 0)]
    for root in scene["nodes"]:
        visit(root)
    node_ids = {old: new for new, old in enumerate(kept)}
    model["nodes"] = [copy.deepcopy(nodes[i]) for i in kept]
    for node in model["nodes"]:
        if "children" in node:
            node["children"] = [node_ids[i] for i in node["children"] if i in node_ids]
    model["scenes"] = [
        {
            "name": "Booster" if booster else "Upper stage",
            "nodes": [node_ids[i] for i in scene["nodes"] if i in node_ids],
        }
    ]
    model["scene"] = 0
    mesh_ids = sorted({node["mesh"] for node in model["nodes"] if "mesh" in node})
    mesh_map = {old: new for new, old in enumerate(mesh_ids)}
    model["meshes"] = [model["meshes"][i] for i in mesh_ids]
    for node in model["nodes"]:
        if "mesh" in node:
            node["mesh"] = mesh_map[node["mesh"]]
    accessor_ids = set()
    for mesh in model["meshes"]:
        for primitive in mesh["primitives"]:
            accessor_ids.update(primitive["attributes"].values())
            if "indices" in primitive:
                accessor_ids.add(primitive["indices"])
            assert not primitive.get("targets")
    accessor_ids = sorted(accessor_ids)
    accessor_map = {old: new for new, old in enumerate(accessor_ids)}
    model["accessors"] = [model["accessors"][i] for i in accessor_ids]
    for mesh in model["meshes"]:
        for primitive in mesh["primitives"]:
            primitive["attributes"] = {
                key: accessor_map[i] for key, i in primitive["attributes"].items()
            }
            if "indices" in primitive:
                primitive["indices"] = accessor_map[primitive["indices"]]
    packed = bytearray()
    new_views = []

    def pack_view(view, start, length):
        packed.extend(b"\0" * (-len(packed) % 4))
        result = dict(view, buffer=0, byteOffset=len(packed), byteLength=length)
        packed.extend(binary[start : start + length])
        new_views.append(result)
        return len(new_views) - 1

    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    sizes = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    for accessor in model["accessors"]:
        assert not accessor.get("sparse")
        view = model["bufferViews"][accessor["bufferView"]]
        assert view["buffer"] == 0
        width = widths[accessor["type"]] * sizes[accessor["componentType"]]
        length = (accessor["count"] - 1) * view.get("byteStride", width) + width
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        accessor["bufferView"] = pack_view(view, start, length)
        accessor["byteOffset"] = 0
    for image in model.get("images", []):
        view = model["bufferViews"][image["bufferView"]]
        image["bufferView"] = pack_view(view, view.get("byteOffset", 0), view["byteLength"])
    model["bufferViews"] = new_views
    model["buffers"] = [{"byteLength": len(packed)}]
    encoded = json.dumps(model, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    packed.extend(b"\0" * (-len(packed) % 4))
    size = 12 + 8 + len(encoded) + 8 + len(packed)
    return (
        struct.pack("<4sIII4s", b"glTF", 2, size, len(encoded), b"JSON")
        + encoded
        + struct.pack("<I4s", len(packed), b"BIN\0")
        + packed
    )


if __name__ == "__main__":
    assets = Path(__file__).resolve().parents[2] / "assets"
    source = (assets / "falcon9.glb").read_bytes()
    for name, booster in [("falcon9-booster.glb", True), ("falcon9-upper-stage.glb", False)]:
        data = split_model(source, booster=booster)
        (assets / name).write_bytes(data)
        print(f"{name}: {len(data):,} bytes")
