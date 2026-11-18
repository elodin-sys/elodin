"""Stage extraction preserves the reference model's geometry and alignment."""

import json
import struct
from pathlib import Path

from export_stage_models import split_model


def document(data):
    length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20 : 20 + length])


def accessor_bytes(data, model, index):
    accessor = model["accessors"][index]
    view = model["bufferViews"][accessor["bufferView"]]
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    sizes = {5121: 1, 5123: 2, 5125: 4, 5126: 4}
    width = widths[accessor["type"]] * sizes[accessor["componentType"]]
    length = (accessor["count"] - 1) * view.get("byteStride", width) + width
    start = (
        28
        + struct.unpack_from("<I", data, 12)[0]
        + view.get("byteOffset", 0)
        + accessor.get("byteOffset", 0)
    )
    return data[start : start + length]


def test_stage_partition_preserves_model_origin_and_geometry():
    source = (Path(__file__).resolve().parents[2] / "assets/falcon9.glb").read_bytes()
    original = document(source)
    reference = {node["name"]: node for node in original["nodes"]}
    partitions = []
    for booster in [True, False]:
        output = split_model(source, booster=booster)
        model = document(output)
        names = {node["name"] for node in model["nodes"]}
        assert ("stage_one_f9-v1.2_rig" in names) == booster
        assert ("stage_two_f9-v1.2_rig" in names) != booster
        assert ("left fairing shell" in names) != booster
        assert not model.get("extensionsRequired")
        for node in model["nodes"]:
            old = reference[node["name"]]
            for key in ["matrix", "translation", "rotation", "scale"]:
                assert node.get(key) == old.get(key)
            if "mesh" in node:
                new_mesh = model["meshes"][node["mesh"]]
                old_mesh = original["meshes"][old["mesh"]]
                assert new_mesh["name"] == old_mesh["name"]
                assert len(new_mesh["primitives"]) == len(old_mesh["primitives"])
                for new_primitive, old_primitive in zip(
                    new_mesh["primitives"], old_mesh["primitives"]
                ):
                    for attribute, index in new_primitive["attributes"].items():
                        assert accessor_bytes(output, model, index) == accessor_bytes(
                            source, original, old_primitive["attributes"][attribute]
                        )
                    if "indices" in new_primitive:
                        assert accessor_bytes(
                            output, model, new_primitive["indices"]
                        ) == accessor_bytes(source, original, old_primitive["indices"])
        meshes = {mesh["name"] for mesh in model["meshes"]}
        partitions.append(meshes)
        filename = "falcon9-booster.glb" if booster else "falcon9-upper-stage.glb"
        assert output == (Path(__file__).resolve().parents[2] / "assets" / filename).read_bytes()
    assert not partitions[0] & partitions[1]
    assert partitions[0] | partitions[1] == {mesh["name"] for mesh in original["meshes"]}
