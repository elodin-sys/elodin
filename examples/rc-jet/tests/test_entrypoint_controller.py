"""Check controller registration through the real RC-jet entry point."""

from pathlib import Path
import runpy

import elodin as el
import pytest


@pytest.mark.parametrize(
    ("non_interactive", "explicit_binary", "expected_recipes"),
    [(False, False, 1), (True, False, 0), (True, True, 1)],
)
def test_controller_registration(
    monkeypatch, tmp_path, non_interactive, explicit_binary, expected_recipes
):
    monkeypatch.setenv("ELODIN_RC_JET_TERRAIN", "static")
    monkeypatch.setenv("ELODIN_RC_JET_SCENARIO", "demo")
    monkeypatch.setenv("ELODIN_NON_INTERACTIVE", "1" if non_interactive else "0")
    monkeypatch.delenv("ELODIN_RC_JET_CONTROLLER_BIN", raising=False)
    if explicit_binary:
        binary = tmp_path / "controller"
        binary.touch()
        monkeypatch.setenv("ELODIN_RC_JET_CONTROLLER_BIN", str(binary))
    recipes = []
    runs = []
    monkeypatch.setattr(el.World, "recipe", lambda self, recipe: recipes.append(recipe))
    monkeypatch.setattr(el.World, "run", lambda self, *args, **kwargs: runs.append(kwargs))

    runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"))

    assert len(recipes) == expected_recipes
    assert len(runs) == 1
    assert runs[0]["interactive"] is not non_interactive
