"""BDX aero-package loader.

Validates and exposes the open-air generated `elodin_package` (schema 1.0).
After `load()` succeeds the package is the only source of aircraft constants;
aircraft numbers must not be restated in Python (improvement guide §9.1).
All rejection rules run before world creation and raise `PackageError`.

The sim world frame (ECEF) intentionally differs from the package's
`frames.world = ENU_Z_UP` string: coefficients are consumed in the *body*
frame through the declared adapter, which is world-agnostic. The package
string documents the local-level convention the adapter was specified
against; it is validated verbatim and recorded as upstream feedback
(see ai-context/bdx/openair_integration_feedback.md).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"
CONCEPT = "bdx"
PHASE = "baseline"

DEFAULT_PACKAGE_DIR = Path(__file__).parent / "model" / "elodin_package"

CREDIBILITY_ORDER = (
    "geometry-correlated",
    "analysis-correlated",
    "ground-test-correlated",
    "flight-correlated",
)

# Simulation modes and their package-tier requirements (guide §8.3 rule 6).
MODE_LONGITUDINAL = "longitudinal_package"
MODE_CLASS_D_6DOF = "class_d_augmented_6dof"
MODE_PACKAGE_6DOF = "package_6dof"

SUPPORTED_FRAMES = {
    "world": "ENU_Z_UP",
    "body": "X_FORWARD_Y_LEFT_Z_UP",
    "geometry": "X_NOSE_TO_TAIL_Y_RIGHT_Z_UP",
    "coefficient_source": "STANDARD_AEROSPACE_X_FORWARD_Y_RIGHT_Z_DOWN",
    "glb": "X_FORWARD_Y_LEFT_Z_UP_CG_ORIGIN",
    "moment_reference": "CG",
}

COEFFICIENTS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")
DERIVATIVE_STATE_KEYS = frozenset({"alpha", "beta", "p", "q", "r", "mach", "u"})

_TOP_LEVEL_KEYS = frozenset(
    {
        "aero",
        "concept",
        "created_at",
        "credibility",
        "frames",
        "limits",
        "manifest",
        "mass_properties",
        "model_id",
        "performance_anchors",
        "phase",
        "propulsion",
        "provenance",
        "reference_geometry",
        "schema_version",
        "trim_map_asset",
        "validity",
    }
)
_FRAMES_KEYS = frozenset(
    {
        "body",
        "body_origin_in_geometry_m",
        "coefficient_adapter",
        "coefficient_source",
        "geometry",
        "geometry_to_body_matrix",
        "glb",
        "moment_reference",
        "world",
    }
)
_AERO_KEYS = frozenset(
    {
        "allowances",
        "derivative_source",
        "derivatives",
        "drag_polar_fit",
        "linearization",
        "polar_asset",
        "validity_component_required",
    }
)
_VALIDITY_KEYS = frozenset(
    {
        "attached_flow_alpha_deg",
        "derivatives_local_only",
        "extrapolation_policy",
        "mach",
        "notes",
        "polar_table_alpha_deg",
        "reynolds_per_m",
    }
)
_MASS_KEYS = frozenset(
    {
        "cg_body_m",
        "cg_geometry_m",
        "cg_z_source",
        "diagonal_approximation_declared",
        "elodin_diagonal_kg_m2",
        "evidence_class",
        "fuel_capacity_kg",
        "fuel_mass_kg",
        "fuel_volume_m3",
        "full_inertia_tensor_kg_m2",
        "inertia_source",
        "manufacturer_listed_mass_kg",
        "manufacturer_mass_state",
        "mass_kg",
        "operating_empty_mass_kg",
        "reserve_fuel_kg",
    }
)
_PROPULSION_KEYS = frozenset(
    {
        "evidence_class",
        "map_asset",
        "model",
        "provisional",
        "provisional_reason",
        "thrust_application_body_m",
        "thrust_axis_body",
    }
)
_MANIFEST_ENTRY_KEYS = frozenset({"path", "role", "sha256", "size_bytes"})


class PackageError(ValueError):
    """The package failed contract validation; the world must not be created."""


def _check_keys(block: dict, expected: frozenset, where: str) -> None:
    keys = set(block)
    missing = expected - keys
    unknown = keys - expected
    if missing:
        raise PackageError(f"{where}: missing fields {sorted(missing)}")
    if unknown:
        raise PackageError(f"{where}: unknown fields {sorted(unknown)} (schema {SCHEMA_VERSION})")


def _require(block: dict, keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in block]
    if missing:
        raise PackageError(f"{where}: missing fields {missing}")


@dataclass(frozen=True)
class Linearization:
    cl0: float
    cl_alpha_per_rad: float
    cm0: float
    cm_alpha_per_rad: float
    trim_control: str
    trim_control_value_deg: float
    reference_alpha_deg: float
    reference_airspeed_mps: float
    reference_altitude_m: float
    reference_cg_x_m: float


@dataclass(frozen=True)
class DragPolar:
    cd0: float
    k: float
    cl_domain: tuple[float, float]


@dataclass(frozen=True)
class Derivatives:
    """Normalized derivative tier (absent for the current BDX baseline).

    `controls` is keyed by producer mixing ids (e.g. "elevator", or
    "collective_elevon" for elevon aircraft) — names are dynamic by contract.
    """

    coefficient_reference: dict[str, str]
    base: dict[str, float]
    state: dict[str, dict[str, float]]
    controls: dict[str, dict[str, float]]


@dataclass(frozen=True)
class Aero:
    linearization: Linearization
    drag_polar: DragPolar
    derivatives: Derivatives | None
    validity_component_required: bool


@dataclass(frozen=True)
class ReferenceGeometry:
    area_m2: float
    span_m: float
    mac_m: float
    aspect_ratio: float


@dataclass(frozen=True)
class MassProperties:
    mass_kg: float
    operating_empty_mass_kg: float
    fuel_mass_kg: float
    fuel_capacity_kg: float
    reserve_fuel_kg: float
    cg_geometry_m: tuple[float, float, float]
    inertia_diagonal_kg_m2: tuple[float, float, float] | None
    manufacturer_listed_mass_kg: tuple[float, float]


@dataclass(frozen=True)
class Propulsion:
    max_thrust_sl_n: float
    min_throttle: float
    dry_mass_kg: float
    fuel_flow_max_kg_s: float
    thrust_application_body_m: tuple[float, float, float]
    thrust_axis_body: tuple[float, float, float]
    provisional: bool


@dataclass(frozen=True)
class PropulsionMap:
    """Regular (altitude, mach, throttle) grid of steady thrust and fuel flow."""

    altitudes_m: np.ndarray
    machs: np.ndarray
    throttles: np.ndarray
    thrust_n: np.ndarray
    fuel_flow_kg_s: np.ndarray


@dataclass(frozen=True)
class TrimRow:
    condition: str
    altitude_m: float
    tas_mps: float
    mach: float
    alpha_deg: float
    beta_deg: float
    control_name: str
    control_deg: float
    throttle: float
    fuel_flow_kg_s: float


@dataclass(frozen=True)
class Validity:
    mach: tuple[float, float]
    attached_flow_alpha_deg: tuple[float, float]
    polar_table_alpha_deg: tuple[float, float]
    extrapolation_policy: str


@dataclass(frozen=True)
class AeroTables:
    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray


@dataclass(frozen=True)
class Frames:
    world: str
    body: str
    geometry: str
    coefficient_source: str
    glb: str
    moment_reference: str
    geometry_to_body_matrix: np.ndarray


@dataclass(frozen=True)
class BdxModel:
    package_dir: Path
    model_id: str
    credibility: str
    frames: Frames
    reference: ReferenceGeometry
    mass: MassProperties
    aero: Aero
    propulsion: Propulsion
    propulsion_map: PropulsionMap
    trim_rows: dict[str, TrimRow]
    validity: Validity
    tables: AeroTables
    performance_anchors: dict
    limits: dict
    provenance: dict
    glb_path: Path

    def require_mode(self, mode: str, allow_class_d: bool = False) -> None:
        """Refuse modes whose required package tier is absent (rule 6)."""
        if mode == MODE_LONGITUDINAL:
            return
        if mode == MODE_PACKAGE_6DOF:
            if self.aero.derivatives is None:
                raise PackageError(
                    "mode package_6dof requires aero.derivatives, which this package "
                    "does not provide (null is evidence of absence)"
                )
            if self.mass.inertia_diagonal_kg_m2 is None:
                raise PackageError("mode package_6dof requires a package inertia tensor")
            return
        if mode == MODE_CLASS_D_6DOF:
            if not allow_class_d:
                raise PackageError(
                    "mode class_d_augmented_6dof requires an explicit scenario opt-in "
                    "(allow_class_d=True)"
                )
            return
        raise PackageError(f"unknown simulation mode {mode!r}")

    def require_credibility(self, minimum: str) -> None:
        if minimum not in CREDIBILITY_ORDER:
            raise PackageError(f"unknown credibility level {minimum!r}")
        have = CREDIBILITY_ORDER.index(self.credibility)
        need = CREDIBILITY_ORDER.index(minimum)
        if have < need:
            raise PackageError(
                f"package credibility {self.credibility!r} is below the scenario "
                f"minimum {minimum!r}"
            )


def _verify_manifest(package_dir: Path, manifest: dict) -> None:
    root = package_dir.resolve()
    for name, entry in sorted(manifest.items()):
        if not isinstance(entry, dict):
            raise PackageError(f"manifest[{name}]: entry must be an object")
        _check_keys(entry, _MANIFEST_ENTRY_KEYS, f"manifest[{name}]")
        rel = Path(entry["path"])
        if rel.is_absolute():
            raise PackageError(f"manifest[{name}]: absolute path {rel}")
        if ".." in rel.parts:
            raise PackageError(f"manifest[{name}]: path escapes the package: {rel}")
        target = package_dir / rel
        # Reject symlinks anywhere along the in-package part of the path.
        probe = package_dir
        for part in rel.parts:
            probe = probe / part
            if probe.is_symlink():
                raise PackageError(f"manifest[{name}]: {probe} is a symlink")
        resolved = target.resolve()
        if root != resolved and root not in resolved.parents:
            raise PackageError(f"manifest[{name}]: resolved path leaves the package: {resolved}")
        if not target.is_file():
            raise PackageError(
                f"manifest[{name}]: missing file {rel} (if this is a fresh checkout, "
                "run `git lfs pull`)"
            )
        size = target.stat().st_size
        if size != entry["size_bytes"]:
            raise PackageError(
                f"manifest[{name}]: size mismatch for {rel}: {size} != {entry['size_bytes']} "
                "(a git-lfs pointer file would also fail this check; run `git lfs pull`)"
            )
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if sha != entry["sha256"]:
            raise PackageError(f"manifest[{name}]: SHA-256 mismatch for {rel}")


def _validate_frames(frames: dict, cg_geometry: tuple[float, float, float]) -> Frames:
    _check_keys(frames, _FRAMES_KEYS, "frames")
    for key, expected in SUPPORTED_FRAMES.items():
        if frames[key] != expected:
            raise PackageError(
                f"frames.{key} = {frames[key]!r} differs from the supported adapter ({expected!r})"
            )
    matrix = np.asarray(frames["geometry_to_body_matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise PackageError("frames.geometry_to_body_matrix must be 4x4")
    cg_x, cg_y, cg_z = cg_geometry
    expected_matrix = np.array(
        [
            [-1.0, 0.0, 0.0, cg_x],
            [0.0, -1.0, 0.0, -cg_y],
            [0.0, 0.0, 1.0, -cg_z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    if not np.allclose(matrix, expected_matrix, atol=1e-9):
        raise PackageError(
            "frames.geometry_to_body_matrix does not match the supported "
            "x_b = cg_x - x_g / y_b = -y_g / z_b = z_g - cg_z adapter"
        )
    return Frames(
        world=frames["world"],
        body=frames["body"],
        geometry=frames["geometry"],
        coefficient_source=frames["coefficient_source"],
        glb=frames["glb"],
        moment_reference=frames["moment_reference"],
        geometry_to_body_matrix=matrix,
    )


def _parse_derivatives(raw: dict | None) -> Derivatives | None:
    if raw is None:
        return None
    _require(raw, ("coefficient_reference", "base", "state", "controls"), "aero.derivatives")
    base = raw["base"]
    if set(base) != set(COEFFICIENTS):
        raise PackageError(f"aero.derivatives.base must have keys {COEFFICIENTS}")
    state = raw["state"]
    if set(state) != set(COEFFICIENTS):
        raise PackageError(f"aero.derivatives.state must have keys {COEFFICIENTS}")
    for coef, terms in state.items():
        unknown = set(terms) - DERIVATIVE_STATE_KEYS
        if unknown:
            raise PackageError(f"aero.derivatives.state[{coef}]: unknown keys {sorted(unknown)}")
    controls = raw["controls"]
    if not controls:
        raise PackageError("aero.derivatives.controls must not be empty")
    for group, terms in controls.items():
        if set(terms) != set(COEFFICIENTS):
            raise PackageError(f"aero.derivatives.controls[{group}] must have keys {COEFFICIENTS}")
    return Derivatives(
        coefficient_reference={k: str(v) for k, v in raw["coefficient_reference"].items()},
        base={k: float(v) for k, v in base.items()},
        state={k: {s: float(v) for s, v in terms.items()} for k, terms in state.items()},
        controls={g: {k: float(v) for k, v in terms.items()} for g, terms in controls.items()},
    )


def _load_propulsion_map(path: Path) -> PropulsionMap:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise PackageError(f"{path.name}: empty propulsion map")
    points: dict[tuple[float, float, float], tuple[float, float]] = {}
    for row in rows:
        key = (float(row["altitude_m"]), float(row["mach"]), float(row["throttle"]))
        points[key] = (float(row["thrust_n"]), float(row["fuel_flow_kg_s"]))
    altitudes = np.array(sorted({k[0] for k in points}))
    machs = np.array(sorted({k[1] for k in points}))
    throttles = np.array(sorted({k[2] for k in points}))
    shape = (altitudes.size, machs.size, throttles.size)
    if len(points) != int(np.prod(shape)):
        raise PackageError(f"{path.name}: propulsion map grid is not full-factorial")
    thrust = np.empty(shape)
    fuel_flow = np.empty(shape)
    for i, alt in enumerate(altitudes):
        for j, mach in enumerate(machs):
            for k, thr in enumerate(throttles):
                thrust[i, j, k], fuel_flow[i, j, k] = points[(alt, mach, thr)]
    return PropulsionMap(
        altitudes_m=altitudes,
        machs=machs,
        throttles=throttles,
        thrust_n=thrust,
        fuel_flow_kg_s=fuel_flow,
    )


def _load_trim_map(path: Path) -> dict[str, TrimRow]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    trim: dict[str, TrimRow] = {}
    for row in rows:
        if row["valid"].strip().lower() != "true":
            raise PackageError(f"{path.name}: trim row {row['condition']!r} is marked invalid")
        trim[row["condition"]] = TrimRow(
            condition=row["condition"],
            altitude_m=float(row["altitude_m"]),
            tas_mps=float(row["tas_mps"]),
            mach=float(row["mach"]),
            alpha_deg=float(row["alpha_deg"]),
            beta_deg=float(row["beta_deg"]),
            control_name=row["control_name"],
            control_deg=float(row["control_deg"]),
            throttle=float(row["throttle"]),
            fuel_flow_kg_s=float(row["fuel_flow_kg_s"]),
        )
    for condition in ("cruise", "dash"):
        if condition not in trim:
            raise PackageError(f"{path.name}: missing required trim row {condition!r}")
    return trim


def _load_tables(path: Path, model_id: str, phase: str) -> AeroTables:
    with np.load(path, allow_pickle=False) as data:
        for key, expected in (
            ("schema_version", SCHEMA_VERSION),
            ("model_id", model_id),
            ("phase", phase),
        ):
            got = str(data[key].item() if data[key].shape == () else data[key])
            if got != expected:
                raise PackageError(f"{path.name}: identity {key}={got!r}, expected {expected!r}")
        alpha = np.asarray(data["alpha_deg"], dtype=np.float64)
        tables = AeroTables(
            alpha_deg=alpha,
            cl=np.asarray(data["CL"], dtype=np.float64),
            cd=np.asarray(data["CD"], dtype=np.float64),
            cm=np.asarray(data["Cm"], dtype=np.float64),
        )
    if not (tables.alpha_deg.size and np.all(np.diff(tables.alpha_deg) > 0)):
        raise PackageError(f"{path.name}: alpha_deg must be strictly increasing")
    for name in ("cl", "cd", "cm"):
        if getattr(tables, name).shape != tables.alpha_deg.shape:
            raise PackageError(f"{path.name}: {name} shape mismatch")
    return tables


def load(package_dir: str | Path | None = None) -> BdxModel:
    """Load and validate the vendored BDX package; raise PackageError on any breach."""
    package_dir = Path(package_dir or os.environ.get("ELODIN_BDX_PACKAGE") or DEFAULT_PACKAGE_DIR)
    model_path = package_dir / "elodin_model.json"
    if not model_path.is_file():
        raise PackageError(f"no elodin_model.json in {package_dir}")
    raw = json.loads(model_path.read_text())

    _check_keys(raw, _TOP_LEVEL_KEYS, "elodin_model.json")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise PackageError(
            f"unsupported schema_version {raw['schema_version']!r}, expected {SCHEMA_VERSION!r}"
        )
    if raw["concept"] != CONCEPT or raw["phase"] != PHASE:
        raise PackageError(
            f"package identity {raw['concept']!r}/{raw['phase']!r} does not match the "
            f"requested {CONCEPT!r}/{PHASE!r}"
        )
    if raw["credibility"] not in CREDIBILITY_ORDER:
        raise PackageError(f"unknown credibility {raw['credibility']!r}")

    _verify_manifest(package_dir, raw["manifest"])

    mass_raw = raw["mass_properties"]
    _check_keys(mass_raw, _MASS_KEYS, "mass_properties")
    cg_geometry = tuple(float(v) for v in mass_raw["cg_geometry_m"])
    frames = _validate_frames(raw["frames"], cg_geometry)

    aero_raw = raw["aero"]
    _check_keys(aero_raw, _AERO_KEYS, "aero")
    lin_raw = aero_raw["linearization"]
    _require(
        lin_raw,
        (
            "CL0",
            "CL_alpha_per_rad",
            "Cm0",
            "Cm_alpha_per_rad",
            "trim_control",
            "trim_control_value_deg",
            "reference_alpha_deg",
            "reference_airspeed_mps",
            "reference_altitude_m",
            "reference_cg_x_m",
        ),
        "aero.linearization",
    )
    linearization = Linearization(
        cl0=float(lin_raw["CL0"]),
        cl_alpha_per_rad=float(lin_raw["CL_alpha_per_rad"]),
        cm0=float(lin_raw["Cm0"]),
        cm_alpha_per_rad=float(lin_raw["Cm_alpha_per_rad"]),
        trim_control=str(lin_raw["trim_control"]),
        trim_control_value_deg=float(lin_raw["trim_control_value_deg"]),
        reference_alpha_deg=float(lin_raw["reference_alpha_deg"]),
        reference_airspeed_mps=float(lin_raw["reference_airspeed_mps"]),
        reference_altitude_m=float(lin_raw["reference_altitude_m"]),
        reference_cg_x_m=float(lin_raw["reference_cg_x_m"]),
    )
    polar_raw = aero_raw["drag_polar_fit"]
    _require(polar_raw, ("CD0", "k", "CL_domain", "equation"), "aero.drag_polar_fit")
    if polar_raw["equation"] != "CD = CD0 + k * CL^2":
        raise PackageError(f"unsupported drag polar equation {polar_raw['equation']!r}")
    drag_polar = DragPolar(
        cd0=float(polar_raw["CD0"]),
        k=float(polar_raw["k"]),
        cl_domain=tuple(float(v) for v in polar_raw["CL_domain"]),
    )
    aero = Aero(
        linearization=linearization,
        drag_polar=drag_polar,
        derivatives=_parse_derivatives(aero_raw["derivatives"]),
        validity_component_required=bool(aero_raw["validity_component_required"]),
    )

    validity_raw = raw["validity"]
    _check_keys(validity_raw, _VALIDITY_KEYS, "validity")
    validity = Validity(
        mach=tuple(float(v) for v in validity_raw["mach"]),
        attached_flow_alpha_deg=tuple(float(v) for v in validity_raw["attached_flow_alpha_deg"]),
        polar_table_alpha_deg=tuple(float(v) for v in validity_raw["polar_table_alpha_deg"]),
        extrapolation_policy=str(validity_raw["extrapolation_policy"]),
    )
    if validity.extrapolation_policy != "flag_invalid_do_not_clamp":
        raise PackageError(f"unsupported extrapolation policy {validity.extrapolation_policy!r}")

    inertia = mass_raw["elodin_diagonal_kg_m2"]
    mass = MassProperties(
        mass_kg=float(mass_raw["mass_kg"]),
        operating_empty_mass_kg=float(mass_raw["operating_empty_mass_kg"]),
        fuel_mass_kg=float(mass_raw["fuel_mass_kg"]),
        fuel_capacity_kg=float(mass_raw["fuel_capacity_kg"]),
        reserve_fuel_kg=float(mass_raw["reserve_fuel_kg"]),
        cg_geometry_m=cg_geometry,
        inertia_diagonal_kg_m2=(tuple(float(v) for v in inertia) if inertia is not None else None),
        manufacturer_listed_mass_kg=tuple(
            float(v) for v in mass_raw["manufacturer_listed_mass_kg"]
        ),
    )

    ref_raw = raw["reference_geometry"]
    _require(ref_raw, ("area_m2", "span_m", "mac_m", "aspect_ratio"), "reference_geometry")
    reference = ReferenceGeometry(
        area_m2=float(ref_raw["area_m2"]),
        span_m=float(ref_raw["span_m"]),
        mac_m=float(ref_raw["mac_m"]),
        aspect_ratio=float(ref_raw["aspect_ratio"]),
    )

    prop_raw = raw["propulsion"]
    _check_keys(prop_raw, _PROPULSION_KEYS, "propulsion")
    model_raw = prop_raw["model"]
    application = prop_raw["thrust_application_body_m"]
    # x_m is null in the current package (engine station unmeasured): treat the
    # thrust point as CG-station with the declared vertical offset.
    propulsion = Propulsion(
        max_thrust_sl_n=float(model_raw["max_thrust_sl_n"]),
        min_throttle=float(model_raw["min_throttle"]),
        dry_mass_kg=float(model_raw["dry_mass_kg"]),
        fuel_flow_max_kg_s=float(model_raw["fuel_flow_max_kg_s"]),
        thrust_application_body_m=(
            float(application["x_m"] or 0.0),
            float(application["y_m"] or 0.0),
            float(application["z_m"] or 0.0),
        ),
        thrust_axis_body=tuple(float(v) for v in prop_raw["thrust_axis_body"]),
        provisional=bool(prop_raw["provisional"]),
    )

    anchors = raw["performance_anchors"]
    _require(anchors, ("cruise", "dash", "stall", "positive_g", "negative_g"), "anchors")
    provenance = raw["provenance"]
    _require(provenance, ("design_sha256", "pipeline_run_id", "evidence_classes"), "provenance")

    return BdxModel(
        package_dir=package_dir,
        model_id=str(raw["model_id"]),
        credibility=str(raw["credibility"]),
        frames=frames,
        reference=reference,
        mass=mass,
        aero=aero,
        propulsion=propulsion,
        propulsion_map=_load_propulsion_map(package_dir / raw["propulsion"]["map_asset"]),
        trim_rows=_load_trim_map(package_dir / raw["trim_map_asset"]),
        validity=validity,
        tables=_load_tables(
            package_dir / aero_raw["polar_asset"], str(raw["model_id"]), str(raw["phase"])
        ),
        performance_anchors=anchors,
        limits=raw["limits"],
        provenance=provenance,
        glb_path=package_dir / raw["manifest"]["render_glb"]["path"],
    )
