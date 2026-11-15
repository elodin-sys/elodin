"""Production vision code cannot reach synthetic truth or world gate poses."""

import ast
from pathlib import Path

import pytest
from controls import GuidanceUpdate

_EXAMPLE = Path(__file__).resolve().parents[1]
_VISION_MODULES = ("vision.py", "gate_detector.py")
_FORBIDDEN_FOR_VISION = {
    "course",
    "referee",
    "referee_audit",
    "hold_referee",
    "sim",
    "main",
    "synthetic_vision",
    "replay",
}
_SYNTHETIC_IMPORTERS = {"synthetic_vision.py", "evaluate_detector.py"}
_TRUTH_FIELDS = {
    "world_pos",
    "world_vel",
    "gate_pose",
    "gate_poses",
    "truth_state",
    "detection",
    "detections",
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module.split(".", maxsplit=1)[0])
    return names


@pytest.mark.parametrize("module_name", _VISION_MODULES)
def test_vision_module_cannot_import_truth_sources(module_name: str) -> None:
    imported = _imported_modules(_EXAMPLE / module_name)

    assert imported.isdisjoint(_FORBIDDEN_FOR_VISION)


def test_only_gate_detector_imports_cv2() -> None:
    offenders = []
    for path in _EXAMPLE.glob("*.py"):
        if path.name == "gate_detector.py":
            continue
        if "cv2" in _imported_modules(path):
            offenders.append(path.name)
    assert offenders == []
    assert "cv2" in _imported_modules(_EXAMPLE / "gate_detector.py")


def test_replay_does_not_import_detector_or_truth_scoring() -> None:
    imported = _imported_modules(_EXAMPLE / "replay.py")
    forbidden = {
        "cv2",
        "gate_detector",
        "course",
        "referee",
        "referee_audit",
        "hold_referee",
        "synthetic_vision",
    }

    assert imported.isdisjoint(forbidden)


def test_only_the_synthetic_generator_imports_synthetic_vision() -> None:
    offenders = []
    for path in _EXAMPLE.glob("*.py"):
        if path.name in _SYNTHETIC_IMPORTERS:
            continue
        if "synthetic_vision" in _imported_modules(path):
            offenders.append(path.name)
    assert offenders == []


def test_guidance_update_has_no_truth_or_detection_fields() -> None:
    names = {field.name for field in GuidanceUpdate.__dataclass_fields__.values()}

    assert names.isdisjoint(_TRUTH_FIELDS)
