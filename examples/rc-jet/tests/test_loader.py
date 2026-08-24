"""Package loader contract tests (improvement guide §8.3 / §9.7).

Anchor values are read from the package at test time, never copied into
literals (guide §9.7).
"""

import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

import bdx_model
from bdx_model import (
    MODE_CLASS_D_6DOF,
    MODE_LONGITUDINAL,
    MODE_PACKAGE_6DOF,
    PackageError,
)


@pytest.fixture()
def package_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "elodin_package"
    shutil.copytree(bdx_model.DEFAULT_PACKAGE_DIR, dst)
    return dst


def mutate_model(package: Path, fn) -> None:
    path = package / "elodin_model.json"
    raw = json.loads(path.read_text())
    fn(raw)
    path.write_text(json.dumps(raw))


def test_load_happy_path():
    model = bdx_model.load()
    raw = json.loads((bdx_model.DEFAULT_PACKAGE_DIR / "elodin_model.json").read_text())
    lin = raw["aero"]["linearization"]
    assert model.aero.linearization.cl_alpha_per_rad == lin["CL_alpha_per_rad"]
    assert model.aero.linearization.cm_alpha_per_rad == lin["Cm_alpha_per_rad"]
    assert model.reference.area_m2 == raw["reference_geometry"]["area_m2"]
    assert model.mass.mass_kg == raw["mass_properties"]["mass_kg"]
    assert model.aero.derivatives is None
    assert model.mass.inertia_diagonal_kg_m2 is None
    assert model.trim_rows["cruise"].altitude_m == 300.0
    assert model.glb_path.is_file()
    # Static stability and drag polar sanity, asserted on the loaded config itself.
    assert model.aero.linearization.cm_alpha_per_rad < 0
    assert model.aero.linearization.cl_alpha_per_rad > 0
    assert model.aero.drag_polar.cd0 > 0 and model.aero.drag_polar.k > 0


def test_reject_wrong_schema(package_copy):
    mutate_model(package_copy, lambda raw: raw.update(schema_version="2.0"))
    with pytest.raises(PackageError, match="schema_version"):
        bdx_model.load(package_copy)


def test_reject_wrong_identity(package_copy):
    mutate_model(package_copy, lambda raw: raw.update(phase="optimized"))
    with pytest.raises(PackageError, match="identity"):
        bdx_model.load(package_copy)


def test_reject_unknown_top_level_field(package_copy):
    mutate_model(package_copy, lambda raw: raw.update(surprise=1))
    with pytest.raises(PackageError, match="unknown fields"):
        bdx_model.load(package_copy)


def test_reject_corrupt_sidecar(package_copy):
    path = package_copy / "trim_map.csv"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(PackageError, match="SHA-256 mismatch"):
        bdx_model.load(package_copy)


def test_reject_size_mismatch(package_copy):
    path = package_copy / "propulsion_map.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(PackageError, match="size mismatch"):
        bdx_model.load(package_copy)


def test_reject_missing_file(package_copy):
    (package_copy / "aero_tables.npz").unlink()
    with pytest.raises(PackageError, match="missing file"):
        bdx_model.load(package_copy)


def test_reject_absolute_manifest_path(package_copy):
    def mutate(raw):
        raw["manifest"]["trim_map"]["path"] = str(package_copy / "trim_map.csv")

    mutate_model(package_copy, mutate)
    with pytest.raises(PackageError, match="absolute path"):
        bdx_model.load(package_copy)


def test_reject_escaping_manifest_path(package_copy):
    outside = package_copy.parent / "outside.csv"
    shutil.copy(package_copy / "trim_map.csv", outside)

    def mutate(raw):
        raw["manifest"]["trim_map"]["path"] = "../outside.csv"

    mutate_model(package_copy, mutate)
    with pytest.raises(PackageError, match="escapes the package"):
        bdx_model.load(package_copy)


def test_reject_symlinked_manifest_entry(package_copy):
    target = package_copy.parent / "linked_trim.csv"
    shutil.move(package_copy / "trim_map.csv", target)
    (package_copy / "trim_map.csv").symlink_to(target)
    with pytest.raises(PackageError, match="symlink"):
        bdx_model.load(package_copy)


def test_reject_frame_mismatch(package_copy):
    def mutate(raw):
        raw["frames"]["body"] = "X_FORWARD_Y_RIGHT_Z_DOWN"

    mutate_model(package_copy, mutate)
    with pytest.raises(PackageError, match="frames.body"):
        bdx_model.load(package_copy)


def test_reject_tampered_geometry_matrix(package_copy):
    def mutate(raw):
        raw["frames"]["geometry_to_body_matrix"][0][3] += 0.5

    mutate_model(package_copy, mutate)
    with pytest.raises(PackageError, match="geometry_to_body_matrix"):
        bdx_model.load(package_copy)


def test_reject_invalid_trim_row(package_copy):
    path = package_copy / "trim_map.csv"
    text = path.read_text().replace("true", "false")
    path.write_bytes(text.encode())

    def mutate(raw):
        entry = raw["manifest"]["trim_map"]
        entry["size_bytes"] = (package_copy / "trim_map.csv").stat().st_size
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    mutate_model(package_copy, mutate)
    with pytest.raises(PackageError, match="marked invalid"):
        bdx_model.load(package_copy)


def test_mode_requirements():
    model = bdx_model.load()
    model.require_mode(MODE_LONGITUDINAL)
    with pytest.raises(PackageError, match="derivatives"):
        model.require_mode(MODE_PACKAGE_6DOF)
    with pytest.raises(PackageError, match="opt-in"):
        model.require_mode(MODE_CLASS_D_6DOF)
    model.require_mode(MODE_CLASS_D_6DOF, allow_class_d=True)
    with pytest.raises(PackageError, match="unknown simulation mode"):
        model.require_mode("warp_drive")


def test_credibility_floor():
    model = bdx_model.load()
    model.require_credibility("analysis-correlated")
    model.require_credibility("geometry-correlated")
    with pytest.raises(PackageError, match="below the scenario minimum"):
        model.require_credibility("flight-correlated")


def synthetic_derivatives() -> dict:
    coefficients = ("CL", "CD", "CY", "Cl", "Cm", "Cn")
    return {
        "coefficient_reference": {"rates": "p*b/(2V), q*c/(2V), r*b/(2V)", "angles": "radians"},
        "base": {c: 0.01 for c in coefficients},
        "state": {
            c: {"alpha": 0.1, "beta": -0.05, "p": -0.4, "q": -8.0, "r": 0.1} for c in coefficients
        },
        "controls": {
            "elevator": {c: 0.05 for c in coefficients},
            "aileron": {c: 0.02 for c in coefficients},
            "rudder": {c: 0.01 for c in coefficients},
        },
    }


def test_forward_compatible_derivatives(package_copy):
    mutate_model(
        package_copy,
        lambda raw: raw["aero"].update(
            derivatives=synthetic_derivatives(), derivative_source="flightdyn.json .derivatives"
        ),
    )
    model = bdx_model.load(package_copy)
    derivatives = model.aero.derivatives
    assert derivatives is not None
    # Control groups are dynamic mixing ids, preserved by name.
    assert set(derivatives.controls) == {"elevator", "aileron", "rudder"}
    assert derivatives.state["Cm"]["q"] == -8.0


def test_malformed_derivatives_rejected(package_copy):
    broken = synthetic_derivatives()
    del broken["base"]["CL"]
    mutate_model(package_copy, lambda raw: raw["aero"].update(derivatives=broken))
    with pytest.raises(PackageError, match="derivatives.base"):
        bdx_model.load(package_copy)


def test_glb_contract():
    """Named nodes, metre units, CG-origin body frame (guide §9.7)."""
    model = bdx_model.load()
    with model.glb_path.open("rb") as f:
        magic, _version, _length = struct.unpack("<III", f.read(12))
        assert magic == 0x46546C67  # 'glTF'
        chunk_len, chunk_type = struct.unpack("<II", f.read(8))
        assert chunk_type == 0x4E4F534A  # JSON
        gltf = json.loads(f.read(chunk_len))
    names = {node["name"] for node in gltf["nodes"]}
    assert names == {"fuselage", "wing", "htail", "fin_c"}
    for node in gltf["nodes"]:
        assert "scale" not in node and "translation" not in node and "rotation" not in node
    extras = gltf["scenes"][0]["extras"]
    assert extras["frame"] == model.frames.glb
    assert extras["units"] == "m"


def test_propulsion_map_grid():
    model = bdx_model.load()
    grid = model.propulsion_map
    assert np.all(np.diff(grid.altitudes_m) > 0)
    assert np.all(np.diff(grid.machs) > 0)
    assert np.all(np.diff(grid.throttles) > 0)
    assert grid.throttles[0] == model.propulsion.min_throttle
    # Thrust and fuel flow strictly increase with throttle at every grid node.
    assert np.all(np.diff(grid.thrust_n, axis=2) > 0)
    assert np.all(np.diff(grid.fuel_flow_kg_s, axis=2) > 0)


def test_polar_table_consistency():
    """The npz sweep (fixed tail, incidence 0) is internally consistent with the
    fitted drag polar and shows the same stability character as the trim-set
    linearization. Pointwise CL/Cm equality is NOT expected: the linearization
    embeds the -1.221 deg trim incidence while the table is an incidence-0 sweep
    (guide §3.3)."""
    model = bdx_model.load()
    lin = model.aero.linearization
    polar = model.aero.drag_polar
    cd_fit = polar.cd0 + polar.k * model.tables.cl**2
    assert np.allclose(cd_fit, model.tables.cd, atol=1e-3)
    alpha = np.deg2rad(model.tables.alpha_deg)
    table_cl_slope = np.polyfit(alpha, model.tables.cl, 1)[0]
    table_cm_slope = np.polyfit(alpha, model.tables.cm, 1)[0]
    assert table_cl_slope == pytest.approx(lin.cl_alpha_per_rad, rel=0.10)
    assert table_cm_slope < 0 and lin.cm_alpha_per_rad < 0
    assert np.all(np.diff(model.tables.cl) > 0)
    assert np.all(np.diff(model.tables.cm) < 0)
