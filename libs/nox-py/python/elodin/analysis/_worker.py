"""Run one analysis script and write its envelope: ``python -m elodin.analysis._worker``.

Locally ``elodin mcp execute`` sends the spec on stdin and reads ``envelope.json`` from the
output directory. In an Elodin Cloud Job the spec comes from ``$ELODIN_EXEC_URL/v1/exec/spec``
with ``$ELODIN_EXEC_TOKEN``, and artifacts and the envelope are posted back to the same API.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import contextlib  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import urllib.request  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

MAX_RESULT = 16 * 1024
MAX_LOG = 1 << 20
MAX_MESSAGE = 2048
SCRIPT = "script.py"


class _Capped(io.StringIO):
    """Keeps the first ``MAX_LOG`` characters written."""

    def write(self, s):
        room = MAX_LOG - self.tell()
        if room > 0:
            super().write(s[:room])
        return len(s)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _version() -> str:
    try:
        return metadata.version("elodin")
    except metadata.PackageNotFoundError:
        return "unknown"


def _error(exc: BaseException, code: str) -> dict:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == SCRIPT]
    line = frames[-1].lineno if frames else None
    lines = code.splitlines()
    err = {
        "type": type(exc).__name__,
        "message": str(exc)[:MAX_MESSAGE],
        "line": line,
        "source_line": lines[line - 1].strip() if line and line <= len(lines) else None,
    }
    if getattr(exc, "did_you_mean", None):
        err["did_you_mean"] = exc.did_you_mean
    return err


def run(spec: dict, backend, out: Path) -> dict:
    """Execute ``spec['code']`` with ``el`` bound to ``backend``; returns the envelope."""
    from elodin import analysis as el

    out.mkdir(parents=True, exist_ok=True)
    code, params = spec["code"], spec.get("parameters") or {}
    limits = spec.get("limits") or {}
    el._bind(backend, str(out), limits.get("max_rows"))
    warnings = []
    cli = os.environ.get("ELODIN_CLI_VERSION")
    if cli and cli != _version():
        warnings.append(f"elodin CLI {cli} differs from the elodin package {_version()}")

    namespace = {"__name__": "__main__", "el": el, "ctx": el.Context(params)}
    stdout, stderr = _Capped(), _Capped()
    started = time.monotonic()
    status, result, error = "completed", None, None
    try:
        compiled = compile(code, SCRIPT, "exec")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compiled, namespace)
        if "result" not in namespace:
            raise NameError("the script must assign a JSON-serializable dict to `result`")
        result = el._jsonable(namespace["result"])
        size = len(json.dumps(result))
        if size > MAX_RESULT:
            raise ValueError(
                f"result is {size:,} bytes; keep it under {MAX_RESULT:,} and save the rest as artifacts"
            )
    except BaseException as exc:  # noqa: BLE001 - every script failure becomes an envelope
        status, result = "error", None
        error = _error(exc, code)
        if isinstance(exc, NameError) and "result" in str(exc):
            error["type"] = "NoResult"
        (out / "traceback.txt").write_text("".join(traceback.format_exception(exc)))
    wall_ms = int((time.monotonic() - started) * 1000)

    for name, buf in (("stdout.txt", stdout), ("stderr.txt", stderr)):
        if buf.tell():
            (out / name).write_text(buf.getvalue())
    files = [
        p
        for p in sorted(out.iterdir())
        if p.is_file() and p.name not in ("code.py", "parameters.json", "envelope.json")
    ]
    artifact_list = []
    for p in files:
        data = p.read_bytes()
        artifact_list.append(
            {"name": p.name, "path": str(p), "bytes": len(data), "sha256": _sha256(data)}
        )

    return {
        "status": status,
        "execution_id": spec.get("execution_id"),
        "result": result,
        "artifacts": artifact_list,
        "warnings": warnings,
        "error": error,
        "provenance": {
            "code_sha256": _sha256(code.encode()),
            "parameters_sha256": _sha256(json.dumps(params, sort_keys=True).encode()),
            "lake_snapshot": spec.get("lake_snapshot"),
            "sources": [{"run": r["name"], "source_id": r.get("source_id")} for r in backend.runs],
            "elodin": _version(),
            "python": platform.python_version(),
        },
        "usage": {"wall_ms": wall_ms, "rows_read": el._state["rows_read"]},
    }


def _local() -> None:
    from elodin.analysis import _LocalBackend

    spec = json.load(sys.stdin)
    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "code.py").write_text(spec["code"])
    (out / "parameters.json").write_text(
        json.dumps(spec.get("parameters") or {}, indent=2, sort_keys=True)
    )
    envelope = run(spec, _LocalBackend(spec["sources"]), out)
    (out / "envelope.json").write_text(json.dumps(envelope, indent=2))


def _request(
    url: str,
    token: str,
    method: str = "GET",
    data: bytes | None = None,
    ctype: str = "application/json",
):
    req = urllib.request.Request(
        url, data, {"authorization": f"Bearer {token}", "content-type": ctype}, method=method
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def _cloud(url: str, token: str) -> None:
    from elodin.analysis import _CloudBackend

    url = url.rstrip("/")
    spec = json.loads(_request(f"{url}/v1/exec/spec", token))
    out = Path(os.environ.get("ELODIN_EXEC_OUT", "/scratch/artifacts"))
    envelope = run(spec, _CloudBackend(url, token, spec), out)
    for a in envelope["artifacts"]:
        path = Path(a.pop("path"))
        _request(
            f"{url}/v1/exec/artifacts/{a['name']}",
            token,
            "PUT",
            path.read_bytes(),
            "application/octet-stream",
        )
    _request(f"{url}/v1/exec/result", token, "POST", json.dumps(envelope).encode())


def main() -> None:
    url, token = os.environ.get("ELODIN_EXEC_URL"), os.environ.get("ELODIN_EXEC_TOKEN")
    if url and token:
        _cloud(url, token)
    else:
        _local()


if __name__ == "__main__":
    main()
