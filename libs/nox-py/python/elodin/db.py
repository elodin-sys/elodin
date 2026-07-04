"""Elodin-DB client: write and read telemetry from plain Python.

Quick start::

    import elodin.db as edb

    server = edb.Server.start("./mydb")            # or connect to `elodin-db run`
    client = edb.Client.connect("127.0.0.1:2240")

    writer = client.table_writer({
        "drone.imu.accel": edb.f64[3].labeled("x", "y", "z"),
        "drone.cmd.throttle": edb.f64,
    })
    writer.write(timestamp_us=t_us, values={
        "drone.imu.accel": [0.0, 0.0, -9.81],
        "drone.cmd.throttle": 0.42,
    })

    ts, accel = client.time_series("drone.imu.accel", t0_us, t1_us)
    table = client.sql("SELECT * FROM 'drone.imu.accel'")   # pyarrow.Table

Design notes:

* Every ``TableWriter.write`` emits exactly one Impeller2 ``Table`` packet:
  a shared little-endian ``i64`` microsecond timestamp followed by each
  field's values at a fixed, naturally-aligned offset.
* ``write_nowait`` never blocks and never raises for transport reasons; rows
  are dropped (counted in ``writer.dropped``) when the queue is full or the
  database is unreachable.
* All fields declared in a writer's schema are required on every write.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .elodin import db as _native

ComponentInfo = _native.ComponentInfo
ComponentData = _native.ComponentData

_PRIM_NP = {
    "f64": np.dtype("<f8"),
    "f32": np.dtype("<f4"),
    "i64": np.dtype("<i8"),
    "i32": np.dtype("<i4"),
    "i16": np.dtype("<i2"),
    "i8": np.dtype("i1"),
    "u64": np.dtype("<u8"),
    "u32": np.dtype("<u4"),
    "u16": np.dtype("<u2"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("?"),
}


@dataclass(frozen=True)
class Field:
    """A component's dtype + shape + optional element labels.

    Use the module-level instances and index/label them::

        edb.f64            # scalar
        edb.f32[3]         # vector of 3
        edb.f64[3, 3]      # rank-2 tensor
        edb.f64[3].labeled("x", "y", "z")
    """

    prim: str
    shape: Tuple[int, ...] = ()
    element_names: Tuple[str, ...] = field(default_factory=tuple)

    def __getitem__(self, dims) -> "Field":
        if isinstance(dims, int):
            dims = (dims,)
        dims = tuple(int(d) for d in dims)
        if len(dims) > 3:
            raise ValueError("shapes up to rank 3 are supported")
        if any(d <= 0 for d in dims):
            raise ValueError(f"invalid shape {dims}")
        return replace(self, shape=dims)

    def labeled(self, *names: str) -> "Field":
        count = self.count
        if len(names) != count:
            raise ValueError(f"{len(names)} labels for {count} elements")
        return replace(self, element_names=tuple(names))

    @property
    def dtype(self) -> np.dtype:
        return _PRIM_NP[self.prim]

    @property
    def count(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.count * self.dtype.itemsize


f64 = Field("f64")
f32 = Field("f32")
i64 = Field("i64")
i32 = Field("i32")
i16 = Field("i16")
i8 = Field("i8")
u64 = Field("u64")
u32 = Field("u32")
u16 = Field("u16")
u8 = Field("u8")
bool_ = Field("bool")


class _PackedField:
    __slots__ = ("name", "dtype", "count", "offset", "nbytes")

    def __init__(self, name: str, spec: Field, offset: int):
        self.name = name
        self.dtype = spec.dtype
        self.count = spec.count
        self.offset = offset
        self.nbytes = spec.nbytes


class TableWriter:
    """Batched writer: one Table packet per row, shared microsecond timestamp."""

    def __init__(
        self,
        addr: str,
        schema: Dict[str, Field],
        queue: str = "drop",
        maxlen: int = 1024,
    ):
        if queue not in ("drop", "drop-oldest", "drop-newest"):
            raise ValueError(f"unknown queue policy {queue!r}")
        if not schema:
            raise ValueError("schema must contain at least one component")
        self._fields: list[_PackedField] = []
        native_fields = []
        offset = 8  # i64 timestamp occupies bytes 0..8
        for name, spec in schema.items():
            if not isinstance(spec, Field):
                raise TypeError(f"schema[{name!r}] must be a Field (e.g. edb.f64[3])")
            align = spec.dtype.itemsize
            offset = (offset + align - 1) // align * align
            self._fields.append(_PackedField(name, spec, offset))
            native_fields.append(
                (
                    name,
                    spec.prim,
                    [int(d) for d in spec.shape],
                    ",".join(spec.element_names) if spec.element_names else None,
                    offset,
                    spec.nbytes,
                )
            )
            offset += spec.nbytes
        self._row_size = offset
        self._ts_struct = struct.Struct("<q")
        self._w = _native.TableWriter(addr, native_fields, maxlen=maxlen)

    @property
    def dropped(self) -> int:
        """Rows dropped by write_nowait (queue full / DB unreachable)."""
        return self._w.dropped

    @property
    def row_size(self) -> int:
        return self._w.row_size

    def _pack(self, timestamp_us: int, values: Dict[str, Any]) -> bytes:
        buf = bytearray(self._row_size)
        self._ts_struct.pack_into(buf, 0, timestamp_us)
        for f in self._fields:
            try:
                v = values[f.name]
            except KeyError:
                missing = [g.name for g in self._fields if g.name not in values]
                raise ValueError(f"write requires all declared fields; missing {missing}") from None
            arr = np.asarray(v, dtype=f.dtype)
            if arr.size != f.count:
                raise ValueError(f"{f.name}: expected {f.count} elements, got {arr.size}")
            buf[f.offset : f.offset + f.nbytes] = arr.tobytes()
        return bytes(buf)

    def write(self, timestamp_us: int, values: Dict[str, Any]) -> None:
        """Blocking write; raises if the row cannot be handed to the socket."""
        self._w.write_row(self._pack(timestamp_us, values))

    def write_nowait(self, timestamp_us: int, values: Dict[str, Any]) -> None:
        """Non-blocking write; drops (never raises) when the queue is full or
        the database is down. See ``dropped``."""
        self._w.write_row_nowait(self._pack(timestamp_us, values))

    def close(self) -> None:
        self._w.close()

    def __enter__(self) -> "TableWriter":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def sql_table_name(component_name: str) -> str:
    """The DataFusion table name elodin-db derives from a component name
    (snake_case with non-alphanumeric characters replaced by underscores)."""
    import re

    s = component_name
    # camelCase / PascalCase boundaries -> underscore (mirrors convert_case)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    return s.lower()


@dataclass(frozen=True)
class Sample:
    """One component sample from ``Client.latest``."""

    name: str
    timestamp_us: int
    values: np.ndarray


class Client:
    """Client for a running Elodin-DB (external or ``Server.start``-ed)."""

    def __init__(self, addr: str):
        self._addr = addr
        self._c = _native.Client(addr)
        self._send_writers: Dict[str, TableWriter] = {}

    @classmethod
    def connect(cls, addr: str, reconnect: str = "backoff") -> "Client":
        """``reconnect`` is accepted for forward compatibility; writers and the
        latest-value subscription always reconnect with exponential backoff."""
        if reconnect not in ("backoff", "none"):
            raise ValueError(f"unknown reconnect policy {reconnect!r}")
        return cls(addr)

    @property
    def addr(self) -> str:
        return self._addr

    # ── write ────────────────────────────────────────────────────────────
    def table_writer(
        self,
        schema: Dict[str, Field],
        queue: str = "drop",
        maxlen: int = 1024,
    ) -> TableWriter:
        return TableWriter(self._addr, schema, queue=queue, maxlen=maxlen)

    def send(self, name: str, values: Any, timestamp_us: int) -> None:
        """Convenience single-component F64 write (one writer cached per name).
        Prefer ``table_writer`` for hot loops."""
        w = self._send_writers.get(name)
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if w is None:
            spec = f64[len(arr)] if arr.size > 1 else f64
            w = self.table_writer({name: spec})
            self._send_writers[name] = w
        w.write(timestamp_us, {name: arr})

    # ── read ─────────────────────────────────────────────────────────────
    def components(self) -> Dict[str, ComponentInfo]:
        """All components registered in the database (name → ComponentInfo)."""
        return self._c.components()

    def latest(self, name: str) -> Optional[Sample]:
        """Latest sample seen on the real-time stream (starts a background
        subscription on first call; may return None until data flows)."""
        data = self._c.latest(name)
        if data is None:
            return None
        return Sample(
            name=data.name,
            timestamp_us=data.timestamp,
            values=np.asarray(data.values),
        )

    def time_series(
        self,
        name: str,
        start_us: int,
        stop_us: int,
        limit: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Historical samples of ``name`` in ``[start_us, stop_us)``.

        Returns ``(timestamps, values)``: ``timestamps`` is int64 microseconds
        of shape (N,), ``values`` has shape ``(N, *component_shape)``.
        """
        ts_bytes, data_bytes, prim, dims = self._c.time_series(
            name, int(start_us), int(stop_us), limit
        )
        timestamps = np.frombuffer(ts_bytes, dtype=np.int64)
        values = np.frombuffer(data_bytes, dtype=_PRIM_NP[prim])
        if dims:
            values = values.reshape(-1, *[int(d) for d in dims])
        return timestamps, values

    def sql(self, query: str):
        """Run a DataFusion SQL query; returns a ``pyarrow.Table``.

        Component time series are exposed as tables named by
        ``sql_table_name(component_name)`` (e.g. ``drone.imu.accel`` →
        ``drone_imu_accel``), each with a ``time`` column plus one column per
        element. A ``<name>_stream`` variant exists for streaming queries.
        """
        import pyarrow as pa
        import pyarrow.ipc

        batches = self._c.sql(query)
        tables = [pa.ipc.open_stream(b).read_all() for b in batches]
        if not tables:
            return pa.table({})
        return pa.concat_tables(tables)

    def state(self) -> str:
        """Connection state of the latest-value subscription."""
        return self._c.state()

    def close(self) -> None:
        for w in self._send_writers.values():
            w.close()
        self._send_writers.clear()
        self._c.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return f"Client(addr='{self._addr}')"


class Server:
    """Embedded Elodin-DB server (same engine as ``elodin-db run``)."""

    def __init__(self, native):
        self._s = native

    @staticmethod
    def start(path: str, addr: str = "127.0.0.1:2240") -> "Server":
        """Bind ``addr`` (errors raise immediately, e.g. port in use) and serve
        the database at ``path`` until ``stop()`` or process exit."""
        return Server(_native.Server(str(path), addr))

    @property
    def addr(self) -> str:
        return self._s.addr

    @property
    def path(self) -> str:
        return self._s.path

    def stop(self) -> None:
        self._s.stop()

    def __enter__(self) -> "Server":
        return self

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    def __repr__(self) -> str:
        return f"Server(path='{self.path}', addr='{self.addr}')"


__all__ = [
    "Client",
    "Server",
    "TableWriter",
    "Field",
    "Sample",
    "ComponentInfo",
    "ComponentData",
    "sql_table_name",
    "f64",
    "f32",
    "i64",
    "i32",
    "i16",
    "i8",
    "u64",
    "u32",
    "u16",
    "u8",
    "bool_",
]
