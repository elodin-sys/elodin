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
    table = client.sql(f"SELECT * FROM {edb.sql_table_name('drone.imu.accel')}")

    for row in client.stream(["drone.imu.accel"]):          # live rows
        ...
    client.send_msg("race.collision", {"id": 1}, timestamp_us=t_us)  # events

Design notes:

* Every ``TableWriter.write`` emits exactly one Impeller2 ``Table`` packet:
  a shared little-endian ``i64`` timestamp (microseconds by default,
  nanoseconds with ``timestamp="ns"``) followed by each field's values at a
  fixed, naturally-aligned offset.
* ``write_nowait`` never blocks and never raises for transport reasons; on
  overflow or an unreachable database rows are shed per the queue policy
  (``drop-oldest`` by default, counted in ``writer.dropped``).
* All fields declared in a writer's schema are required on every write.
* Message payloads are opaque bytes on the wire; ``send_msg``/``get_msgs``
  offer a v1 convenience encoding (bytes pass-through, str as UTF-8,
  everything else JSON) — postcard-schema payload inference is deferred.
* Set ``ELODIN_DB_LOG=debug`` to surface native client/server diagnostics.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .elodin import db as _native

ComponentInfo = _native.ComponentInfo

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
    """Batched writer: one Table packet per row, shared timestamp.

    ``queue`` controls what ``write_nowait`` sheds when the bounded queue is
    full: ``"drop-oldest"`` (default) evicts the oldest queued row so fresh
    telemetry wins; ``"drop-newest"`` discards the incoming row.

    ``timestamp`` selects the shared timestamp unit: ``"us"`` (default,
    ``write(timestamp_us=...)``) or ``"ns"`` for nanosecond sources
    (``write(timestamp_ns=...)``; the database stores microseconds).
    """

    def __init__(
        self,
        addr: str,
        schema: Dict[str, Field],
        queue: str = "drop-oldest",
        maxlen: int = 1024,
        timestamp: str = "us",
    ):
        if queue not in ("drop-oldest", "drop-newest"):
            raise ValueError(f"unknown queue policy {queue!r}")
        if timestamp not in ("us", "ns"):
            raise ValueError(f"unknown timestamp unit {timestamp!r}")
        self._ts_unit = timestamp
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
        self._ts_struct = struct.Struct("<q")
        self._w = _native.TableWriter(
            addr, native_fields, maxlen=maxlen, queue=queue, timestamp_unit=timestamp
        )
        # Single source of truth for the packed row size is the native writer.
        self._row_size = self._w.row_size

    @property
    def dropped(self) -> int:
        """Rows dropped by write_nowait (queue full / DB unreachable)."""
        return self._w.dropped

    @property
    def last_error(self) -> Optional[str]:
        """Most recent transport error or database rejection (e.g. a schema
        mismatch for an already-registered component), or None."""
        return self._w.last_error

    def state(self) -> str:
        """Writer connection state: "Connected" | "Disconnected"."""
        return self._w.state()

    @property
    def row_size(self) -> int:
        return self._w.row_size

    def _timestamp(self, timestamp_us: Optional[int], timestamp_ns: Optional[int]) -> int:
        if (timestamp_us is None) == (timestamp_ns is None):
            raise TypeError("pass exactly one of timestamp_us / timestamp_ns")
        if self._ts_unit == "us":
            if timestamp_us is None:
                raise TypeError(
                    'writer uses microseconds; pass timestamp_us (or use timestamp="ns")'
                )
            return int(timestamp_us)
        if timestamp_ns is None:
            raise TypeError('writer uses nanoseconds; pass timestamp_ns (or use timestamp="us")')
        return int(timestamp_ns)

    def _pack(self, timestamp: int, values: Dict[str, Any]) -> bytes:
        buf = bytearray(self._row_size)
        self._ts_struct.pack_into(buf, 0, timestamp)
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

    def write(
        self,
        timestamp_us: Optional[int] = None,
        values: Optional[Dict[str, Any]] = None,
        *,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """Blocking write; raises if the row cannot be handed to the socket."""
        if values is None:
            raise TypeError("write requires values")
        self._w.write_row(self._pack(self._timestamp(timestamp_us, timestamp_ns), values))

    def write_nowait(
        self,
        timestamp_us: Optional[int] = None,
        values: Optional[Dict[str, Any]] = None,
        *,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """Non-blocking write; drops (never raises) when the queue is full or
        the database is down. See ``dropped``."""
        if values is None:
            raise TypeError("write_nowait requires values")
        self._w.write_row_nowait(self._pack(self._timestamp(timestamp_us, timestamp_ns), values))

    def close(self) -> None:
        self._w.close()

    def __enter__(self) -> "TableWriter":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def sql_table_name(component_name: str) -> str:
    """The DataFusion table name elodin-db derives from a component name
    (e.g. ``drone.imu.accel`` -> ``drone_imu_accel``). Delegates to the exact
    conversion the database server uses."""
    return _native.sql_table_name(component_name)


@dataclass(frozen=True)
class Sample:
    """One component sample from ``Client.latest``."""

    name: str
    timestamp_us: int
    values: np.ndarray


def _to_array(data: bytes, prim: str, shape) -> np.ndarray:
    values = np.frombuffer(data, dtype=_PRIM_NP[prim])
    if shape:
        return values.reshape(*[int(d) for d in shape])
    return values.reshape(())


@dataclass(frozen=True)
class StreamRow:
    """One row from ``Client.stream``: the requested components present in
    that tick's Table packet.

    ``timestamp_us`` is the newest sample timestamp in the row. A batched
    stream carries each component's *latest* value, so when mixing rates a
    slow component's sample can be older than the row — its own time is in
    ``timestamps[name]``.
    """

    timestamp_us: int
    values: Dict[str, np.ndarray]
    timestamps: Dict[str, int]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.values[name]

    def __contains__(self, name: str) -> bool:
        return name in self.values


class ComponentStream:
    """Iterator over stream rows; ends when the stream is closed or its
    connection fails. Use as a context manager (or call ``close()``) to stop
    the underlying subscription."""

    _POLL_MS = 200

    def __init__(self, native):
        self._s = native

    def __iter__(self) -> "ComponentStream":
        return self

    def __next__(self) -> StreamRow:
        while True:
            row = self._s.next_row(self._POLL_MS)
            if row is not None:
                timestamp_us, parts = row
                values = {}
                timestamps = {}
                for name, data, prim, shape, ts in parts:
                    values[name] = _to_array(data, prim, shape)
                    timestamps[name] = ts
                return StreamRow(timestamp_us=timestamp_us, values=values, timestamps=timestamps)
            if self._s.is_closed():
                raise StopIteration

    def close(self) -> None:
        self._s.close()

    def __enter__(self) -> "ComponentStream":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def _encode_msg_payload(payload: Any) -> bytes:
    """bytes pass through untouched; str is UTF-8; everything else is JSON."""
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode()
    import json

    return json.dumps(payload).encode()


def _decode_msg_payload(data: bytes) -> Any:
    """Inverse convenience: JSON if it parses, raw bytes otherwise."""
    import json

    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return data


class MessageStream:
    """Iterator over ``(timestamp_us, payload)`` from ``Client.msg_stream``;
    delivers new messages only, coalescing bursts to the latest message per
    server wake. Use ``get_msgs`` for lossless history. Use as a context
    manager (or call ``close()``) to stop the underlying subscription."""

    _POLL_MS = 200

    def __init__(self, native, raw: bool):
        self._s = native
        self._raw = raw

    def __iter__(self) -> "MessageStream":
        return self

    def __next__(self):
        while True:
            item = self._s.next_msg(self._POLL_MS)
            if item is not None:
                timestamp_us, payload = item
                return (
                    timestamp_us,
                    payload if self._raw else _decode_msg_payload(payload),
                )
            if self._s.is_closed():
                raise StopIteration

    def close(self) -> None:
        self._s.close()

    def __enter__(self) -> "MessageStream":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


class Client:
    """Client for a running Elodin-DB (external or ``Server.start``-ed)."""

    def __init__(self, addr: str):
        self._addr = addr
        self._c = _native.Client(addr)
        self._send_writers: Dict[str, TableWriter] = {}
        self._registered_msgs: set[str] = set()

    @classmethod
    def connect(cls, addr: str) -> "Client":
        """Writers and the latest-value subscription always reconnect with
        exponential backoff."""
        return cls(addr)

    @property
    def addr(self) -> str:
        return self._addr

    # ── write ────────────────────────────────────────────────────────────
    def table_writer(
        self,
        schema: Dict[str, Field],
        queue: str = "drop-oldest",
        maxlen: int = 1024,
        timestamp: str = "us",
    ) -> TableWriter:
        return TableWriter(self._addr, schema, queue=queue, maxlen=maxlen, timestamp=timestamp)

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

    # ── message logs ─────────────────────────────────────────────────────
    def send_msg(self, name: str, payload: Any, timestamp_us: int) -> None:
        """Append one message to the log named ``name``.

        Payload encoding is a v1 convenience, not a schema system: ``bytes``
        pass through untouched, ``str`` is UTF-8, and anything else is JSON
        (postcard-schema payload inference is deliberately deferred).
        """
        if name not in self._registered_msgs:
            self._c.register_msg(name)
            self._registered_msgs.add(name)
        self._c.send_msg(name, _encode_msg_payload(payload), int(timestamp_us))

    def get_msgs(
        self,
        name: str,
        start_us: int,
        stop_us: int,
        limit: Optional[int] = None,
        raw: bool = False,
    ) -> list:
        """Historical messages of ``name`` as ``[(timestamp_us, payload)]``.

        Bounds follow the database's message-log semantics: the range is
        inclusive of ``stop_us``, and ``start_us`` snaps to the message at or
        before it. Payloads are JSON-decoded when they parse as JSON (pass
        ``raw=True`` for bytes)."""
        msgs = self._c.get_msgs(name, int(start_us), int(stop_us), limit)
        if raw:
            return list(msgs)
        return [(t, _decode_msg_payload(b)) for t, b in msgs]

    def msg_stream(self, name: str, maxlen: int = 1024, raw: bool = False) -> MessageStream:
        """Live stream of new messages on ``name`` as
        ``(timestamp_us, payload)``. Bursts may coalesce to the latest message;
        use ``get_msgs`` for lossless history and payload decoding details."""
        native = _native.MsgStreamSub(self._addr, name, maxlen=maxlen)
        return MessageStream(native, raw)

    # ── read ─────────────────────────────────────────────────────────────
    def components(self) -> Dict[str, ComponentInfo]:
        """All components registered in the database (name → ComponentInfo)."""
        return self._c.components()

    def earliest_timestamp(self) -> int:
        """Earliest data timestamp in the database (microseconds)."""
        return self._c.earliest_timestamp()

    def stream(
        self,
        names,
        rate_hz: Optional[float] = None,
        start=None,
        maxlen: int = 1024,
    ) -> ComponentStream:
        """Iterate rows of the named components.

        * ``rate_hz=None`` (default): live real-time stream of new data.
        * ``rate_hz=<hz>``: fixed-rate replay from ``start`` — ``"earliest"``
          (default), ``"latest"``, or an ``int`` microsecond timestamp.

        Rows carry only the components present in each tick; iteration ends
        when the stream is closed or its connection fails.
        """
        if isinstance(names, str):
            names = [names]
        if start is not None and rate_hz is None:
            raise ValueError("start= requires rate_hz= (fixed-rate replay)")
        initial, initial_us = "earliest", None
        if isinstance(start, (int, np.integer)):
            initial, initial_us = "manual", int(start)
        elif isinstance(start, str):
            initial = start
        native = _native.StreamSub(
            self._addr,
            list(names),
            rate_hz=rate_hz,
            initial=initial,
            initial_us=initial_us,
            maxlen=maxlen,
        )
        return ComponentStream(native)

    def latest(self, name: str) -> Optional[Sample]:
        """Latest sample seen on the real-time stream (starts a background
        subscription on first call; may return None until data flows).

        Values keep their true dtype and shape (an i64 counter comes back as
        int64, a 3x3 tensor as shape (3, 3))."""
        parts = self._c.latest(name)
        if parts is None:
            return None
        timestamp_us, data, prim, shape = parts
        return Sample(name=name, timestamp_us=timestamp_us, values=_to_array(data, prim, shape))

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
    "StreamRow",
    "ComponentStream",
    "MessageStream",
    "ComponentInfo",
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
