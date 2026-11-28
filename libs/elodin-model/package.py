#!/usr/bin/env python3
"""Build static/shared libelodin_model around one export and lay out a CMake package.

    python3 libs/elodin-model/package.py <model_dir> <package_dir>
        [--backend aot|jit] [--cpu native|generic]

`model_dir` is what `JaxSim.export_model` wrote. A C++ project then uses the package with
`find_package(ElodinModel CONFIG REQUIRED)` and `-DElodinModel_DIR=<package_dir>`.

`--cpu` compiles `step.o` here from the model's StableHLO, so a model exported on another OS or
architecture packages on the delivery machine with Rust and Python 3.11 alone.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CRATE = Path(__file__).resolve().parent
MODEL_FILES = ("step.stablehlo.mlir", "step.o", "state.bin", "schema.json")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("model", type=Path, help="directory JaxSim.export_model wrote")
parser.add_argument("package", type=Path, help="package directory to (re)create")
parser.add_argument(
    "--backend",
    choices=["aot", "jit"],
    default="aot",
    help="aot links the model's precompiled step.o; jit compiles its StableHLO at open",
)
parser.add_argument(
    "--cpu",
    choices=["native", "generic"],
    help="compile step.o on this machine: for its CPU's features, or the architecture baseline",
)
args = parser.parse_args()
if args.cpu and args.backend == "jit":
    parser.error("--cpu applies to the aot backend; jit compiles when the library opens")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(*command: str) -> str:
    """What a command prints, or "unknown" when it fails or is not installed."""
    try:
        result = subprocess.run(
            command,
            cwd=CRATE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def object_platform(path: Path) -> tuple[str | None, str] | None:
    """The architecture and OS a relocatable object was compiled for, when recognized."""
    head = path.read_bytes()[:20]
    if head[:4] == b"\x7fELF":
        machine = int.from_bytes(head[18:20], "little")
        return {62: "x86_64", 183: "aarch64"}.get(machine), "Linux"
    if head[:4] == b"\xcf\xfa\xed\xfe":
        cpu = int.from_bytes(head[4:8], "little")
        return {0x01000007: "x86_64", 0x0100000C: "aarch64"}.get(cpu), "Darwin"
    return None


model = args.model
model_manifest = json.loads((model / "model-manifest.json").read_text())
for name, expected in model_manifest["files"].items():
    if name == "step.o" and args.cpu:
        continue
    path = model / name
    if not path.is_file():
        sys.exit(f"model manifest names missing file {path}")
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        sys.exit(f"{path} does not match model-manifest.json")

scratch = tempfile.TemporaryDirectory(prefix="elodin-model-")
if args.cpu:
    workers = json.loads((model / "schema.json").read_text())["workers"]
    compiled = Path(scratch.name) / "model"
    compiled.mkdir()
    for name in ("step.stablehlo.mlir", "state.bin", "schema.json"):
        shutil.copy2(model / name, compiled)
    compile_object = subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--locked",
            "--quiet",
            "-p",
            "cranelift-mlir",
            "--example",
            "compile-object",
            "--",
            str(compiled / "step.stablehlo.mlir"),
            str(compiled / "step.o"),
            "elodin_model_kernel",
            str(workers),
            args.cpu,
        ],
        cwd=CRATE,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    model_manifest["target"] = json.loads(compile_object.stdout)
    model = compiled
else:
    step = model / "step.o"
    if args.backend == "aot":
        if not step.is_file():
            sys.exit(f"{step} is missing; pass --cpu native or --cpu generic to compile it here")
        machine = platform.machine()
        here = ({"arm64": "aarch64"}.get(machine, machine), platform.system())
        built_for = object_platform(step)
        if built_for and built_for != here:
            sys.exit(
                f"{step} is compiled for {built_for[0]} {built_for[1]}, not this {here[0]} "
                f"{here[1]} machine; pass --cpu native or --cpu generic to compile it here"
            )
model_manifest["files"] = {
    name: {"bytes": (model / name).stat().st_size, "sha256": sha256(model / name)}
    for name in MODEL_FILES
    if (model / name).is_file()
}

rustc = command_output("rustc", "-vV")
host = next(
    (line.removeprefix("host: ") for line in rustc.splitlines() if line.startswith("host: ")),
    "unknown",
)
metadata = json.loads(
    subprocess.run(
        [
            "cargo",
            "metadata",
            "--quiet",
            "--format-version=1",
            "--locked",
            "--filter-platform",
            host,
        ],
        cwd=CRATE,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
)
packages = {package["id"]: package for package in metadata["packages"]}
revision = command_output("git", "rev-parse", "HEAD")
if revision != "unknown" and command_output("git", "status", "--porcelain"):
    revision += "-dirty"
package_build = {
    "backend": args.backend,
    "compiled_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "elodin_revision": revision,
    "target": host,
    "rustc": rustc,
    "cargo": command_output("cargo", "--version"),
    "cranelift": next(
        package["version"]
        for package in packages.values()
        if package["name"] == "cranelift-codegen"
    ),
}
build_info = {
    "format": "elodin.model-build",
    "version": 1,
    "model": model_manifest,
    "build": package_build,
}
build_info_path = Path(scratch.name) / "build-info.json"
build_info_path.write_text(json.dumps(build_info, separators=(",", ":")))

env = dict(
    os.environ,
    ELODIN_MODEL_DIR=str(model.resolve()),
    ELODIN_MODEL_BACKEND=args.backend,
    ELODIN_MODEL_BUILD_INFO_PATH=str(build_info_path),
)
build = subprocess.run(
    [
        "cargo",
        "build",
        "--release",
        "--locked",
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
for name in model_manifest["files"]:
    shutil.copy2(model / name, args.package)
(args.package / "model-manifest.json").write_text(json.dumps(model_manifest, indent=2) + "\n")

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
        copy = destination / path.name
        shutil.copy2(path, copy)
        copied.append(str(copy.relative_to(args.package)))
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
