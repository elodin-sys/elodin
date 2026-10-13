"""elodin.analysis: read recordings and save analysis outputs, locally or in Elodin Cloud.

Scripts run by ``elodin mcp execute`` get this module as ``el`` and a ``ctx`` global::

    df = el.signal("mfimumessage.accel", run="flight").window(start_us=a, end_us=b).to_polars()
    result = {"rows": df.height, "table": el.artifacts.save_table(df, "accel")}

Signals and tables follow ``elodin-db export --join --mono-us``: names are lowercase and
match case-insensitively, ``time_us`` is int64 microseconds on the recorder clock, and
private components don't exist. The same script reads a local database (``--db``) or an
Elodin Cloud project (``--project``) and returns the same rows.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.ipc

__all__ = [
    "AmbiguousRun",
    "AnalysisError",
    "Context",
    "ResultBudgetError",
    "Table",
    "UnknownSignal",
    "UnknownTable",
    "artifacts",
    "runs",
    "signal",
    "signals",
    "table",
    "tables",
]

DEFAULT_MAX_ROWS = 5_000_000
PREVIEW_CAP = 100
RESERVED = {
    "code.py",
    "parameters.json",
    "envelope.json",
    "stdout.txt",
    "stderr.txt",
    "traceback.txt",
}


class AnalysisError(Exception):
    """Base class for SDK errors; ``did_you_mean`` lists close names when relevant."""

    did_you_mean: list = []


class UnknownSignal(AnalysisError):
    def __init__(self, message: str, did_you_mean: list):
        super().__init__(message)
        self.did_you_mean = did_you_mean


class UnknownTable(UnknownSignal):
    pass


class AmbiguousRun(AnalysisError):
    pass


class ResultBudgetError(AnalysisError):
    def __init__(self, what: str, run: str, rows: int, max_rows: int):
        super().__init__(
            f"{what} in run {run!r} has {rows:,} rows, over max_rows={max_rows:,}; "
            "narrow it with .window(start_us, end_us) or .select(...), or raise max_rows"
        )
        self.rows, self.max_rows = rows, max_rows


# --- backends -------------------------------------------------------------------------


class _LocalBackend:
    """Reads local databases through the native ``Recording`` reader."""

    def __init__(self, sources: list):
        self._sources = {s["name"]: s for s in sources}
        self._open: dict = {}
        self.runs = []
        for s in sources:
            catalog = s.get("catalog")
            if catalog is None:
                catalog = json.loads(self._recording(s["name"]).catalog_json())
            self.runs.append(
                {
                    "name": s["name"],
                    "id": None,
                    "source_id": s.get("source_id"),
                    "tags": {},
                    "catalog": catalog,
                }
            )

    def _recording(self, run: str):
        if run not in self._open:
            from ..elodin import db as _native

            self._open[run] = _native.Recording.open(self._sources[run]["path"])
        return self._open[run]

    def read(self, run, table, columns, start_us, end_us, nonnull, max_rows):
        data = self._recording(run).read(table, list(columns), start_us, end_us, nonnull)
        return pa.ipc.open_stream(data).read_all()

    def count(self, run, table, start_us, end_us, nonnull):
        return self._recording(run).count(table, start_us, end_us, nonnull)


class _CloudBackend:
    """Reads an Elodin Cloud project through the execution's scoped API."""

    def __init__(self, url: str, token: str, spec: dict):
        self._url, self._token = url.rstrip("/"), token
        self.runs = spec["runs"]

    def _post(self, path: str, body: dict):
        req = urllib.request.Request(
            self._url + path,
            json.dumps(body).encode(),
            {"authorization": f"Bearer {self._token}", "content-type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=600)

    def _query(self, body: dict):
        try:
            return self._post("/v1/exec/query", body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail)
            except ValueError:
                detail = {"error": detail}
            if e.code == 413:
                raise ResultBudgetError(
                    body["table"], body["run"], detail.get("rows", 0), body["max_rows"]
                ) from None
            raise AnalysisError(f"query failed ({e.code}): {detail.get('error', detail)}") from None

    def read(self, run, table, columns, start_us, end_us, nonnull, max_rows):
        body = dict(
            run=run,
            table=table,
            columns=list(columns),
            start_us=start_us,
            end_us=end_us,
            nonnull=nonnull,
            max_rows=max_rows,
        )
        with self._query(body) as r:
            return pa.ipc.open_stream(r.read()).read_all()

    def count(self, run, table, start_us, end_us, nonnull):
        body = dict(
            run=run,
            table=table,
            columns=[],
            start_us=start_us,
            end_us=end_us,
            nonnull=nonnull,
            max_rows=0,
            count=True,
        )
        with self._query(body) as r:
            return json.loads(r.read())["rows"]


_state: dict = {
    "backend": None,
    "out": None,
    "rows_read": 0,
    "max_rows": DEFAULT_MAX_ROWS,
    "artifacts": [],
}


def _bind(backend, out_dir: Optional[str] = None, max_rows: Optional[int] = None):
    """Point the SDK at a backend; the worker calls this before running a script."""
    _state.update(
        backend=backend, out=Path(out_dir) if out_dir else None, rows_read=0, artifacts=[]
    )
    _state["max_rows"] = max_rows or DEFAULT_MAX_ROWS


def open_local(*paths: str, out_dir: Optional[str] = None):
    """Use local databases directly, e.g. from a notebook: ``el.open_local("./flight")``."""
    sources = []
    for p in paths:
        name, _, path = p.rpartition("=") if "=" in p else ("", "", p)
        sources.append({"name": name or Path(path).name, "path": str(Path(path).resolve())})
    _bind(_LocalBackend(sources), out_dir)


def _backend():
    if _state["backend"] is None:
        raise AnalysisError(
            "no recordings bound; run under `elodin mcp execute` or call el.open_local(path)"
        )
    return _state["backend"]


# --- catalog lookups -------------------------------------------------------------------


def _scoped(run: Optional[str]) -> list:
    all_runs = _backend().runs
    if run is None:
        return all_runs
    found = [r for r in all_runs if r["name"].lower() == run.lower()]
    if not found:
        raise AmbiguousRun(f"unknown run {run!r}; runs: {[r['name'] for r in all_runs]}")
    return found


def _one_run(run: Optional[str]) -> dict:
    scoped = _scoped(run)
    if len(scoped) > 1:
        raise AmbiguousRun(
            f"pass run=: {len(scoped)} runs are in scope: {[r['name'] for r in scoped]}"
        )
    return scoped[0]


def _suggest(name: str, kind: str) -> list:
    names: dict = {}
    for r in _backend().runs:
        for item in r["catalog"][kind]:
            rows = item["rows"]
            names.setdefault(item["name"].lower(), {})[r["name"]] = "populated" if rows else "empty"
    q = name.lower()
    close = [n for n in names if q in n or n.rsplit(".", 1)[-1] == q]
    close += [
        n for n in difflib.get_close_matches(q, list(names), n=5, cutoff=0.6) if n not in close
    ]
    out = []
    for n in close[:5]:
        avail = {r["name"]: names[n].get(r["name"], "absent") for r in _backend().runs}
        out.append({"name": n, "runs": avail})
    return out


def _signal_card(run: dict, name: str) -> Optional[dict]:
    for s in run["catalog"]["signals"]:
        if s["name"].lower() == name.lower():
            return s
    return None


def _card(s: dict, run: str) -> dict:
    return {
        "name": s["name"].lower(),
        "dtype": s["dtype"],
        "shape": s["shape"],
        "elements": s["elements"],
        "table": s["table"],
        "column": s["column"],
        "run": run,
        "rows": s["rows"],
        "first_us": s["first_us"],
        "last_us": s["last_us"],
    }


def _matches(query: Optional[str], name: str) -> bool:
    if not query:
        return True
    q, n = query.lower(), name.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", q) if t]
    return q in n or (bool(tokens) and all(t in n for t in tokens))


def runs() -> list:
    """Runs in scope: name, id, source_id, tables, signals, first_us, last_us, tags."""
    out = []
    for r in _backend().runs:
        sig = [s for s in r["catalog"]["signals"] if s["rows"] and not s["timestamp_source"]]
        out.append(
            {
                "name": r["name"],
                "id": r.get("id"),
                "source_id": r.get("source_id"),
                "tags": r.get("tags") or {},
                "tables": sum(1 for t in r["catalog"]["tables"] if t["rows"]),
                "signals": len(r["catalog"]["signals"]),
                "first_us": min((s["first_us"] for s in sig), default=None),
                "last_us": max((s["last_us"] for s in sig), default=None),
            }
        )
    return out


def signals(query: Optional[str] = None, run: Optional[str] = None) -> list:
    """Signal cards whose names match ``query`` (substring or all tokens)."""
    return [
        _card(s, r["name"])
        for r in _scoped(run)
        for s in r["catalog"]["signals"]
        if _matches(query, s["name"])
    ]


def tables(query: Optional[str] = None, run: Optional[str] = None) -> list:
    """Joined export tables whose names match ``query``."""
    return [
        {"name": t["name"], "columns": t["columns"], "rows": t["rows"], "run": r["name"]}
        for r in _scoped(run)
        for t in r["catalog"]["tables"]
        if _matches(query, t["name"])
    ]


def signal(name: str, run: Optional[str] = None) -> "Table":
    """One signal as a Table with columns ``time_us`` and ``value``."""
    if not any(_signal_card(r, name) for r in _scoped(run)):
        raise UnknownSignal(f"no signal named {name!r}", _suggest(name, "signals"))
    r = _one_run(run)
    s = _signal_card(r, name)
    if s is None:
        raise UnknownSignal(
            f"run {r['name']!r} has no signal named {name!r}", _suggest(name, "signals")
        )
    return Table(r, s["table"], [s["column"]], signal=s)


def table(name: str, run: Optional[str] = None) -> "Table":
    """A joined table with ``time_us`` and one column per member (null where absent)."""
    r = _one_run(run)
    for t in r["catalog"]["tables"]:
        if t["name"].lower() == name.lower():
            return Table(r, t["name"], list(t["columns"]))
    raise UnknownTable(f"run {r['name']!r} has no table named {name!r}", _suggest(name, "tables"))


class Table:
    """A lazy view of rows; nothing is read until ``count``, ``to_arrow``, or ``to_polars``."""

    def __init__(
        self,
        run: dict,
        name: str,
        columns: list,
        signal: Optional[dict] = None,
        start_us: Optional[int] = None,
        end_us: Optional[int] = None,
    ):
        self._run, self._name, self._columns, self._signal = run, name, columns, signal
        self._start, self._end = start_us, end_us

    def __repr__(self) -> str:
        what = f"signal {self._signal['name']!r}" if self._signal else f"table {self._name!r}"
        return f"<Table {what} run={self._run['name']!r} window=[{self._start}, {self._end})>"

    def _clone(self, **kw) -> "Table":
        args = dict(
            run=self._run,
            name=self._name,
            columns=self._columns,
            signal=self._signal,
            start_us=self._start,
            end_us=self._end,
        )
        args.update(kw)
        return Table(**args)

    def window(self, start_us: Optional[int] = None, end_us: Optional[int] = None) -> "Table":
        """Rows with ``start_us <= time_us < end_us``, intersected with any earlier window."""
        return self._clone(
            start_us=_tighter(self._start, start_us, max), end_us=_tighter(self._end, end_us, min)
        )

    def select(self, *columns: str) -> "Table":
        """Keep only these member columns; ``time_us`` is always kept."""
        if self._signal:
            raise AnalysisError("select() applies to el.table(...), not a single signal")
        known = {c.lower(): c for c in self._columns}
        missing = [c for c in columns if c.lower() not in known]
        if missing:
            raise UnknownSignal(
                f"table {self._name!r} has no columns {missing}; columns: {self._columns}", []
            )
        return self._clone(columns=[known[c.lower()] for c in columns])

    def _nonnull(self) -> Optional[str]:
        return self._signal["column"] if self._signal else None

    def count(self) -> int:
        """Row count without materializing the data."""
        return _backend().count(
            self._run["name"], self._name, self._start, self._end, self._nonnull()
        )

    def to_arrow(self, max_rows: Optional[int] = None) -> pa.Table:
        """Materialize as a pyarrow.Table; vectors are FixedSizeList."""
        limit = max_rows or _state["max_rows"]
        what = f"signal {self._signal['name']!r}" if self._signal else f"table {self._name!r}"
        rows = self.count()
        if rows > limit:
            raise ResultBudgetError(what, self._run["name"], rows, limit)
        tbl = _backend().read(
            self._run["name"],
            self._name,
            self._columns,
            self._start,
            self._end,
            self._nonnull(),
            limit,
        )
        _state["rows_read"] += tbl.num_rows
        tbl = _fixed_size(tbl.drop_columns(["seq"]), self._run, self._name)
        if self._signal:
            tbl = tbl.rename_columns(["time_us", "value"])
        return tbl

    def to_polars(self, max_rows: Optional[int] = None, flatten: bool = False):
        """Materialize as a polars.DataFrame; ``flatten`` splits vectors into one column per element."""
        import polars as pl

        tbl = self.to_arrow(max_rows)
        if flatten:
            tbl = _flatten(tbl, self._run, self._name, self._signal)
        return pl.from_arrow(tbl)


def _tighter(a, b, pick):
    return b if a is None else a if b is None else pick(a, b)


def _members(run: dict, table_name: str) -> dict:
    return {s["column"]: s for s in run["catalog"]["signals"] if s["table"] == table_name}


def _fixed_size(tbl: pa.Table, run: dict, table_name: str) -> pa.Table:
    """One schema for both backends: nullable fields, vectors as ``fixed_size_list<item>``
    (the lake stores them as variable-size lists)."""
    members = _members(run, table_name)
    fields = []
    for f in tbl.schema:
        s = members.get(f.name)
        t = f.type
        if (
            s
            and s["shape"]
            and (pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t))
        ):
            t = pa.list_(t.value_type, math.prod(s["shape"]))
        fields.append(pa.field(f.name, t, nullable=True))
    return tbl.cast(pa.schema(fields))


def _flatten(tbl: pa.Table, run: dict, table_name: str, sig: Optional[dict]) -> pa.Table:
    members = _members(run, table_name)
    names, cols = [], []
    for f, col in zip(tbl.schema, tbl.columns):
        if not pa.types.is_fixed_size_list(f.type):
            names.append(f.name)
            cols.append(col)
            continue
        s = sig if f.name == "value" and sig else members.get(f.name, {})
        elements = s.get("elements") or []
        flat = col.combine_chunks()
        values = flat.flatten()
        size = f.type.list_size
        for k in range(size):
            suffix = elements[k] if k < len(elements) else str(k)
            names.append(f"{f.name}_{suffix}")
            cols.append(
                values[k::size]
                if flat.null_count == 0
                else pa.array(
                    [None if v is None else v[k] for v in flat.to_pylist()], f.type.value_type
                )
            )
    return pa.table(cols, names=names)


# --- artifacts and context -------------------------------------------------------------


def _artifact_path(name: str, ext: str) -> Path:
    out = _state["out"]
    if out is None:
        raise AnalysisError("artifacts need an output directory; run under `elodin mcp execute`")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
        raise AnalysisError(f"bad artifact name {name!r}: use letters, digits, '.', '_', '-'")
    if not name.endswith(ext):
        name += ext
    if name in RESERVED:
        raise AnalysisError(f"artifact name {name!r} is reserved")
    return out / name


def _record(path: Path) -> dict:
    data = path.read_bytes()
    ref = {"name": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    _state["artifacts"] = [a for a in _state["artifacts"] if a["name"] != path.name] + [ref]
    return dict(ref)


class _Artifacts:
    """Save outputs next to the execution; each call returns ``{name, bytes, sha256}``."""

    def save_table(self, df, name: str) -> dict:
        import pyarrow.parquet as pq

        tbl = df if isinstance(df, pa.Table) else df.to_arrow()
        path = _artifact_path(name, ".parquet")
        pq.write_table(tbl, path)
        return _record(path)

    def save_figure(self, fig, name: str) -> dict:
        path = _artifact_path(name, ".png")
        fig.savefig(path, dpi=100, metadata={"Software": None})
        return _record(path)

    def save_json(self, obj, name: str) -> dict:
        path = _artifact_path(name, ".json")
        path.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True))
        return _record(path)

    def save_html(self, text: str, name: str) -> dict:
        path = _artifact_path(name, ".html")
        path.write_text(text)
        return _record(path)


artifacts = _Artifacts()


class Context:
    """The ``ctx`` global: execution parameters and small previews for ``result``."""

    def __init__(self, parameters: Optional[dict] = None):
        self.parameters = dict(parameters or {})

    def preview(self, df, n: int = 20) -> list:
        """At most ``n`` (cap 100) rows as JSON-ready dicts."""
        n = max(0, min(int(n), PREVIEW_CAP))
        if isinstance(df, pa.Table):
            rows = df.slice(0, n).to_pylist()
        elif hasattr(df, "head") and hasattr(df, "to_dicts"):
            rows = df.head(n).to_dicts()
        else:
            rows = list(df)[:n]
        return _jsonable(rows)


def _jsonable(obj: Any) -> Any:
    """Plain JSON values: numpy scalars and arrays become Python, non-finite floats None."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return _jsonable(obj.tolist())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable")


os.environ.setdefault("MPLBACKEND", "Agg")
