#!/usr/bin/env python3
"""Build libelodin_model around one exported model and lay it out as a CMake package.

    python3 libs/elodin-model/package.py <model_dir> <package_dir> [--backend aot|jit]

`model_dir` is what `JaxSim.export_model` wrote. A C++ project then uses the package with
`find_package(ElodinModel CONFIG REQUIRED)` and `-DElodinModel_DIR=<package_dir>`.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("model", type=Path, help="directory JaxSim.export_model wrote")
parser.add_argument("package", type=Path, help="package directory to (re)create")
parser.add_argument(
    "--backend",
    choices=["aot", "jit"],
    default="aot",
    help="aot links the model's precompiled step.o; jit compiles its StableHLO at open",
)
args = parser.parse_args()

env = dict(
    os.environ,
    ELODIN_MODEL_DIR=str(args.model.resolve()),
    ELODIN_MODEL_BACKEND=args.backend,
)
build = subprocess.run(
    [
        "cargo",
        "build",
        "--release",
        "-p",
        "elodin-model",
        "--message-format=json-render-diagnostics",
    ],
    cwd=CRATE,
    env=env,
    stdout=subprocess.PIPE,
    text=True,
    check=True,
)
library = next(
    (
        Path(path)
        for line in build.stdout.splitlines()
        if (message := json.loads(line)).get("reason") == "compiler-artifact"
        and message["target"]["name"] == "elodin_model"
        for path in message["filenames"]
        if Path(path).suffix in (".so", ".dylib")
    ),
    None,
)
if library is None:
    sys.exit("cargo built no libelodin_model shared library")

shutil.rmtree(args.package, ignore_errors=True)
(args.package / "include").mkdir(parents=True)
(args.package / "lib").mkdir()
shutil.copy2(library, args.package / "lib" / library.name)
shutil.copy2(CRATE / "include" / "elodin_model.h", args.package / "include")
shutil.copy2(CRATE / "ElodinModelConfig.cmake", args.package)
shutil.copy2(args.model / "schema.json", args.package)
print(f"packaged the {args.backend} model library in {args.package}")
