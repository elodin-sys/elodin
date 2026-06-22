"""Monte Carlo campaign helpers for Elodin."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from elodin.elodin import monte_carlo as _native

Param = _native.Param
Params = _native.Params
ParamsSpec = _native.ParamsSpec
params_spec = _native.params_spec
params = _native.params
result = _native.result
port = _native.port
spec_json = _native.spec_json


def export_db(
    db_path: str | Path,
    output: str | Path,
    *,
    format: str = "csv",
    flatten: bool = True,
    join: bool = True,
    mono_ns: bool = True,
    csv_fast_floats: bool = True,
    timeout: float | None = None,
    elodin_db: str | None = None,
) -> Path:
    """Export an Elodin DB into files and return the output directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    binary = elodin_db or os.environ.get("ELODIN_DB_BIN") or shutil.which("elodin-db")
    if binary is None:
        raise RuntimeError("elodin-db binary not found on PATH")
    args = [binary, "export", str(db_path), "--output", str(output), "--format", format]
    if csv_fast_floats:
        args.append("--csv-fast-floats")
    if flatten:
        args.append("--flatten")
    if join:
        args.append("--join")
    if mono_ns:
        args.append("--mono-ns")
    subprocess.run(args, check=True, timeout=timeout)
    return output


__all__ = [
    "Param",
    "Params",
    "ParamsSpec",
    "params_spec",
    "params",
    "result",
    "port",
    "export_db",
    "spec_json",
]
