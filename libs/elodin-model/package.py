#!/usr/bin/env python3
"""Build static/shared libelodin_model around one export and lay out a CMake package.

    python3 libs/elodin-model/package.py <model_dir> <package_dir> [--backend aot|jit]

`model_dir` is what `JaxSim.export_model` wrote. A C++ project then uses the package with
`find_package(ElodinModel CONFIG REQUIRED)` and `-DElodinModel_DIR=<package_dir>`.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(*command: str) -> str:
    result = subprocess.run(
        command,
        cwd=CRATE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


model_manifest_path = args.model / "model-manifest.json"
if model_manifest_path.is_file():
    model_manifest = json.loads(model_manifest_path.read_text())
else:
    model_manifest = {
        "format": "elodin.model-export",
        "version": 1,
        "target": {"cpu": "native"},
        "files": {
            name: {
                "bytes": (args.model / name).stat().st_size,
                "sha256": sha256(args.model / name),
            }
            for name in ("step.stablehlo.mlir", "step.o", "state.bin", "schema.json")
        },
    }
for name, expected in model_manifest.get("files", {}).items():
    path = args.model / name
    if not path.is_file():
        sys.exit(f"model manifest names missing file {path}")
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        sys.exit(f"{path} does not match model-manifest.json")
rustc = command_output("rustc", "-vV")
target = next(
    (line.removeprefix("host: ") for line in rustc.splitlines() if line.startswith("host: ")),
    "unknown",
)
package_build = {
    "backend": args.backend,
    "compiled_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "elodin_revision": command_output("git", "rev-parse", "HEAD"),
    "target": target,
    "rustc": rustc,
    "cargo": command_output("cargo", "--version"),
    "cranelift": "0.130",
}
build_info = {
    "format": "elodin.model-build",
    "version": 1,
    "model": model_manifest,
    "build": package_build,
}
build_info_dir = tempfile.TemporaryDirectory(prefix="elodin-model-build-")
build_info_path = Path(build_info_dir.name) / "build-info.json"
build_info_path.write_text(json.dumps(build_info, separators=(",", ":")))

env = dict(
    os.environ,
    ELODIN_MODEL_DIR=str(args.model.resolve()),
    ELODIN_MODEL_BACKEND=args.backend,
    ELODIN_MODEL_BUILD_INFO_PATH=str(build_info_path),
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
artifacts = [
    Path(path)
    for line in build.stdout.splitlines()
    if (message := json.loads(line)).get("reason") == "compiler-artifact"
    and message["target"]["name"] == "elodin_model"
    for path in message["filenames"]
]
shared = next((path for path in artifacts if path.suffix in (".so", ".dylib")), None)
static = next((path for path in artifacts if path.suffix == ".a"), None)
if shared is None:
    sys.exit("cargo built no libelodin_model shared library")
if static is None:
    sys.exit("cargo built no libelodin_model static library")

shutil.rmtree(args.package, ignore_errors=True)
(args.package / "include").mkdir(parents=True)
(args.package / "lib").mkdir()
shutil.copy2(shared, args.package / "lib" / shared.name)
shutil.copy2(static, args.package / "lib" / static.name)
for header in ("elodin_model.h", "elodin_model.hpp", "elodin_step_hook.h"):
    shutil.copy2(CRATE / "include" / header, args.package / "include")
shutil.copy2(CRATE / "ElodinModelConfig.cmake", args.package)
for name in (
    "schema.json",
    "state.bin",
    "step.stablehlo.mlir",
    "step.o",
    "model-manifest.json",
):
    shutil.copy2(args.model / name, args.package)

metadata = json.loads(
    subprocess.run(
        ["cargo", "metadata", "--format-version=1", "--locked"],
        cwd=CRATE,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
)
packages = {package["id"]: package for package in metadata["packages"]}
nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
root = next(package["id"] for package in metadata["packages"] if package["name"] == "elodin-model")
closure = set()
pending = [root]
while pending:
    package_id = pending.pop()
    if package_id in closure:
        continue
    closure.add(package_id)
    pending.extend(
        dep["pkg"]
        for dep in nodes[package_id]["deps"]
        if any(kind["kind"] is None for kind in dep["dep_kinds"])
    )
runtime_packages = sorted(
    (packages[package_id] for package_id in closure),
    key=lambda package: (package["name"], package["version"]),
)
licenses = args.package / "licenses"
licenses.mkdir()
shutil.copy2(CRATE.parents[1] / "LICENSE", licenses / "Elodin-Apache-2.0.txt")
sbom_packages = []
for package in runtime_packages:
    copied = []
    source_dir = Path(package["manifest_path"]).parent
    destination = licenses / f"{package['name']}-{package['version']}"
    candidates = sorted(
        {
            path
            for pattern in ("LICENSE*", "COPYING*", "NOTICE*")
            for path in source_dir.glob(pattern)
            if path.is_file()
        }
    )
    for path in candidates:
        destination.mkdir(exist_ok=True)
        target = destination / path.name
        shutil.copy2(path, target)
        copied.append(str(target.relative_to(args.package)))
    sbom_packages.append(
        {
            **{key: package.get(key) for key in ("name", "version", "license", "source")},
            "license_files": copied,
        }
    )
sbom = {
    "format": "elodin.runtime-sbom",
    "version": 1,
    "packages": sbom_packages,
}
(args.package / "runtime-sbom.json").write_text(json.dumps(sbom, indent=2) + "\n")

package_files = sorted(
    path
    for path in args.package.rglob("*")
    if path.is_file() and path.name != "build-manifest.json"
)
build_manifest = {
    "format": "elodin.model-package",
    "version": 1,
    "model": model_manifest,
    "build": package_build,
    "files": {
        str(path.relative_to(args.package)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in package_files
    },
}
(args.package / "build-manifest.json").write_text(json.dumps(build_manifest, indent=2) + "\n")
print(f"packaged the {args.backend} model shared and static libraries in {args.package}")
