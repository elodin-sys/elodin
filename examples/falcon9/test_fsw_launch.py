"""Check the real launch recipes without starting a simulation or controller."""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import elodin as el
import pytest


@pytest.mark.parametrize("mode", ["legacy", "aerial", "photo3d"])
@pytest.mark.parametrize("planning", [False, True])
def test_both_launch_paths_pass_the_selected_terrain_mode(monkeypatch, tmp_path, mode, planning):
    main = Path(__file__).with_name("main.py")
    body = ast.parse(main.read_text()).body
    start = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "controller_dir"
            for target in node.targets
        )
    )
    end = next(
        i
        for i, node in enumerate(body[start:], start)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "world"
        and node.value.func.attr == "recipe"
    )
    setup = ast.Module(body=body[start : end + 1], type_ignores=[])
    binary = tmp_path / "controller/target/release/falcon9-fsw"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setenv("ELODIN_MONTE_CARLO_PLANNING", "1" if planning else "0")
    monkeypatch.delenv("ELODIN_MONTE_CARLO_CONTEXT", raising=False)
    recipes = []
    exec(
        compile(setup, str(main), "exec"),
        dict(
            __file__=str(tmp_path / "main.py"),
            Path=Path,
            os=os,
            el=el,
            TERRAIN_MODE=mode,
            mission="crs12",
            DEFAULT_STATE_PORT=9114,
            DEFAULT_COMMAND_PORT=9115,
            world=SimpleNamespace(recipe=recipes.append),
        ),
    )
    assert len(recipes) == 1
    recipe = json.loads(recipes[0].to_json())
    assert recipe["type"] == ("process" if planning else "cargo")
    assert recipe["args"] == ["--terrain-mode", mode]
