"""Exercise the CI launcher while another simulation occupies the default port."""

import errno
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("exit_code", [0, 7])
def test_sensor_camera_uses_a_free_port_and_preserves_failures(tmp_path, exit_code):
    tools = tmp_path / "bin"
    tools.mkdir()

    def executable(name, body):
        path = tools / name
        path.write_text("#!/usr/bin/env python3\n" + body)
        path.chmod(0o755)

    executable(
        "tracy-capture",
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).write_text('trace')\n",
    )
    executable(
        "tracy-csvexport",
        "print('name,mean_ns,count')\n"
        "print('sensor_camera_poll_wait,1000,3')\n"
        "print('sensor_camera_image_copy_driver,1000,3')\n",
    )
    executable("timeout", "import os, sys\nos.execvp(sys.argv[3], sys.argv[3:])\n")
    executable(
        "elodin",
        "import json, os, pathlib, socket, sys\n"
        "args = sys.argv[1:]\n"
        "host, port = args[args.index('--addr') + 1].rsplit(':', 1)\n"
        "sockets = []\n"
        "for adjacent in range(int(port), int(port) + 3):\n"
        "    sock = socket.socket()\n"
        "    sock.bind((host, adjacent))\n"
        "    sockets.append(sock)\n"
        "pathlib.Path(os.environ['ARGS_FILE']).write_text(json.dumps(args))\n"
        "sys.exit(int(os.environ['SIM_EXIT']))\n",
    )
    analysis = tmp_path / "scripts/ci/sensor_camera_tracy_analysis.py"
    analysis.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/ci/sensor_camera_tracy_analysis.py", analysis)
    args_file = tmp_path / "args.json"
    with socket.socket() as busy:
        try:
            busy.bind(("127.0.0.1", 2240))
            busy.listen()
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/ci/sensor_camera_perf.sh")],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
                "ARGS_FILE": str(args_file),
                "SIM_EXIT": str(exit_code),
            },
            capture_output=True,
            text=True,
            timeout=20,
        )
    assert result.returncode == exit_code, result.stdout + result.stderr
    args = json.loads(args_file.read_text())
    assert args[:2] == ["run", "examples/sensor-camera/main.py"]
    address = args[args.index("--addr") + 1]
    assert address.startswith("127.0.0.1:")
    assert address != "127.0.0.1:2240"
