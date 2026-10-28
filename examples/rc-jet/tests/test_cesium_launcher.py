"""Exercise the actual launcher without starting a renderer or contacting ion."""

import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "examples/cesium/run_rc_jet_terrain.sh"


@pytest.fixture
def launch(tmp_path):
    editor = tmp_path / "fake editor"
    editor.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$CESIUM_ION_ASSET_ID" "$ELODIN_RC_JET_HEIGHT_ASSET_ID" '
        '"${CESIUM_ION_IMAGERY_ASSET_ID-unset}" "${CESIUM_TILESET_URL-unset}" "$@"\n'
    )
    editor.chmod(0o755)

    def run(*args, **overrides):
        return subprocess.run(
            ["bash", str(LAUNCHER), *args],
            cwd=REPO_ROOT,
            env={
                "PATH": os.environ["PATH"],
                "IN_NIX_SHELL": "pure",
                "CESIUM_ION_TOKEN": "test-token",
                "ELODIN_EDITOR_BIN": str(editor),
                **overrides,
            },
            capture_output=True,
            text=True,
            check=False,
        )

    return run


@pytest.mark.parametrize("photo_args", [(), ("--photo3d", "on")])
def test_photo3d_replaces_rendering_source_but_not_heights(launch, photo_args):
    result = launch(
        *photo_args,
        "--addr",
        "127.0.0.1:32400",
        CESIUM_ION_ASSET_ID="42",
        CESIUM_ION_IMAGERY_ASSET_ID="2",
        CESIUM_TILESET_URL="https://example.invalid/old-tileset.json",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "2275207",
        "1",
        "unset",
        "unset",
        "editor",
        "examples/rc-jet/main.py",
        "--addr",
        "127.0.0.1:32400",
    ]


def test_photo3d_accepts_an_independent_height_source(launch):
    result = launch("--photo3d", "on", ELODIN_RC_JET_HEIGHT_ASSET_ID="99")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:3] == ["2275207", "99", "unset"]


def test_standard_terrain_preserves_custom_source_and_editor_arguments(launch):
    result = launch(
        "--photo3d", "off", "--", "--schematic", "a layout.py", CESIUM_ION_ASSET_ID="42"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "42",
        "42",
        "2",
        "unset",
        "editor",
        "examples/rc-jet/main.py",
        "--schematic",
        "a layout.py",
    ]


@pytest.mark.parametrize("args", [("--photo3d",), ("--photo3d", "yes")])
def test_invalid_photo3d_mode_does_not_start_an_editor(launch, args):
    result = launch(*args)
    assert result.returncode == 2
    assert "requires 'on' or 'off'" in result.stderr
    assert not result.stdout


def test_missing_token_does_not_start_an_editor(launch):
    result = launch("--photo3d", "on", CESIUM_ION_TOKEN="")
    assert result.returncode == 2
    assert "token missing" in result.stderr
    assert not result.stdout
