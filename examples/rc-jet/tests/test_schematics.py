"""The Python-authored layouts must preserve the existing RC-jet scenes."""

import importlib
import subprocess
import sys
from pathlib import Path

import elodin.ui as ui
import pytest

RC_JET = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("module_name", "source_name"),
    [("bdx_schematic", "bdx.kdl"), ("visual_check_schematic", "visual_check.kdl")],
)
def test_python_schematic_preserves_kdl_model(module_name, source_name):
    module = importlib.import_module(module_name)
    source = ui.from_kdl((RC_JET / source_name).read_text())
    built = module.build()
    assert isinstance(built, ui.Schematic)
    assert ui.from_kdl(built.emit_kdl()) == ui.from_kdl(source.emit_kdl())
    assert module.build().emit_kdl() == built.emit_kdl()

    # Exercise the standalone entry point used by --schematic, without importing
    # main.py or visual_check.py (both start simulations at module scope).
    emitted = subprocess.check_output([sys.executable, str(RC_JET / f"{module_name}.py")])
    assert ui.from_kdl(emitted.decode()) == ui.from_kdl(source.emit_kdl())
