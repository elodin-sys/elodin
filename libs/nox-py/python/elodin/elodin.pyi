from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    Tuple,
    overload,
)

import jax
import numpy
import polars as pl

from elodin import Archetype

class ComponentHandle:
    """One component of a fixed list of entities, resolved by `World.prepare_component`."""

    component_id: int
    name: str
    access: Literal["read", "write"]
    dtype: numpy.dtype
    shape: tuple[int, ...]
    rows: list[EntityId]
    topology_generation: int

class BatchHandle:
    """Handles over the same number of rows, from `World.prepare_batch`."""

    fields: dict[str, ComponentHandle]

class BatchView:
    """A batch's arrays by field name, as attributes or items; also a context manager."""

    def __getattr__(self, name: str) -> numpy.ndarray: ...
    def __getitem__(self, name: str) -> numpy.ndarray: ...
    def __enter__(self) -> BatchView: ...
    def __exit__(self, *exc: object) -> bool: ...

class HistoryHandle:
    """The committed DB history of an entity's component or of a message log, from
    `World.prepare_component_history` or `World.prepare_message_history`."""

    name: str

class TimestampedWriter:
    """Components marked `external_control`, from `World.prepare_timestamped_writer`."""

class WorldStepContext:
    """Passed to a `pre_commit` callback, which runs after the tick and before the commit.

    Views read the world's current values. Writes to a write handle's view reach the world, and
    the same commit, when the callback returns; the systems read them on the next tick. Views
    are copies and end with the callback. History reads see only committed DB samples, and
    deferred writes land after the commit.
    """

    tick: int
    timestamp: int
    def view(self, handle: ComponentHandle) -> numpy.ndarray:
        """The handle's rows, read-only for a read handle."""
        ...
    def batch(self, batch: BatchHandle) -> BatchView:
        """Views every field of `batch`."""
        ...
    def record(self, handle: ComponentHandle) -> None:
        """Records the handle's component, recorded with `Recording.manual()`, at this commit."""
        ...
    def read_at(
        self,
        history: HistoryHandle,
        timestamp: int,
        selection: Literal["exact", "at_or_before"],
    ) -> Optional[tuple[int, numpy.ndarray]]:
        """The committed sample that `selection` picks at `timestamp`: its timestamp and a copy
        of its value, or `None`. A message's value is its bytes as `uint8`."""
        ...
    def defer_write(
        self, writer: TimestampedWriter, timestamp: int, values: dict[str, jax.typing.ArrayLike]
    ) -> None:
        """Appends `values`, one per key of `writer`, at `timestamp` once this cycle's world
        commit lands. Writes keep their call order, across callbacks too."""
        ...
    def witness_bytes(self, name: str, data: jax.typing.ArrayLike) -> None:
        """Fills witness channel `name` with the bytes of `data`: every row of its batch, in
        order. Does nothing when the run records no witness."""
        ...
    def witness_replay(self, name: str) -> Optional[numpy.ndarray]:
        """The recorded bytes of external channel `name`, as `uint8` rows, in a replay that has
        them; otherwise `None`."""
        ...
    def stop_recipes(self) -> None:
        """Gracefully terminate all s10-managed recipes; see `StepContext.stop_recipes`."""
        ...

class Timing:
    """How much a run measures, from least to most perturbing; the receipt reports it."""

    @staticmethod
    def summary() -> Timing:
        """Phase times of every cycle; always on."""
        ...
    @staticmethod
    def systems() -> Timing:
        """Also the time in each compiled function, so each named system, on the Cranelift
        backend; a system small enough to inline counts toward its caller. Its probes cost
        `systems.probe_pair_ns` per call and keep calls serial."""
        ...
    @staticmethod
    def operations() -> Timing:
        """Also sampled time per kind of operation."""
        ...

class NativeStepHook:
    """A `pre_commit` hook in a shared library, over the rows of a prepared batch."""

    @staticmethod
    def load(path: str, input: BatchHandle, symbol: str, workers: int = 1) -> NativeStepHook:
        """Loads `symbol`, and `{symbol}_init` and `{symbol}_destroy` if present, from the
        shared library at `path`. Each tick calls `symbol` once per worker, without the GIL,
        each on a share of the rows."""
        ...

PreCommit = Callable[[int, WorldStepContext], None] | NativeStepHook
"""A `pre_commit` hook; a list of them runs in order."""

class Witness:
    """Batches and hook channels to witness, from `World.prepare_witness`."""

class DeterminismTrace:
    """Where and how a run records its witness, for `World.run(determinism_trace=...)`."""

    path: str
    reference: Optional[str]
    stop_after_first_mismatch: bool
    def __init__(
        self,
        path: str,
        witness: Witness,
        raw: bool | tuple[int, int] = False,
        reference: Optional[str] = None,
        stop_after_first_mismatch: bool = False,
    ) -> None:
        """`raw` keeps the exact bytes of every tick, or of ticks `start..stop`, besides the 16
        ticks on either side of the first mismatch with `reference`."""
        ...

class PrimitiveType:
    F64: PrimitiveType
    U64: PrimitiveType

class Integrator:
    Rk4: Integrator
    SemiImplicit: Integrator
    Rk2: Integrator
    Rk3: Integrator
    Verlet: Integrator
    Rk78: Integrator
    Dp54: Integrator

class StepContext:
    """Context object passed to pre_step and post_step callbacks, providing direct DB read/write access.

    This enables SITL workflows to read sensor data and write component data (like motor
    commands from Betaflight) directly to the database within the same process.
    """
    @property
    def tick(self) -> int:
        """Current simulation tick count."""
        ...
    @property
    def timestamp(self) -> int:
        """Current simulation timestamp (microseconds since epoch)."""
        ...
    def write_component(
        self,
        pair_name: str,
        data: jax.typing.ArrayLike,
        timestamp: Optional[int] = None,
    ) -> None:
        """Write component data to the database.

        Args:
            pair_name: The full component name in "entity.component" format
                      (e.g., "drone.motor_command")
            data: NumPy array containing the component data to write
            timestamp: Optional timestamp (microseconds since epoch) to write at.
                      If None, uses the current simulation timestamp.

        Raises:
            RuntimeError: If the component doesn't exist in the database
            ValueError: If the data size doesn't match the component schema

        Note:
            Timestamps must be monotonically increasing per component. Writing with
            a timestamp less than the last write will raise an error (TimeTravel).
        """
        ...
    def read_component(
        self,
        pair_name: str,
        timestamp: Optional[int] = None,
    ) -> jax.Array:
        """Read component data from the database.

        Args:
            pair_name: The full component name in "entity.component" format
                      (e.g., "drone.accel", "drone.gyro", "drone.world_pos")
            timestamp: Optional timestamp (microseconds since epoch). When None
                       (the default), returns the most recent sample. When
                       provided, returns the sample with the greatest timestamp
                       <= the requested one (floor / sample-and-hold semantics).
                       If the requested timestamp is past the most recent write,
                       the latest sample is returned (clamp-to-latest).

        Returns:
            NumPy array containing the component data (dtype matches component schema).
            The array is always 1D; reshape if needed.

        Raises:
            RuntimeError: If the component doesn't exist, has no data, or the
                          requested timestamp is before the very first sample.
        """
        ...
    def component_batch_operation(
        self,
        reads: list[str] = [],
        writes: Optional[dict[str, jax.typing.ArrayLike]] = None,
        write_timestamps: Optional[dict[str, int]] = None,
        read_timestamps: Optional[dict[str, int]] = None,
    ) -> dict[str, jax.Array]:
        """Perform multiple component reads and writes in a single DB operation.

        This is more efficient than calling read_component/write_component multiple
        times, as it only acquires the database lock once for all operations.

        Args:
            reads: List of component names to read (e.g., ["drone.accel", "drone.gyro"])
            writes: Dict mapping component names to numpy arrays to write
                   (e.g., {"drone.motor_command": motors_array})
            write_timestamps: Optional dict mapping component names to timestamps
                             (microseconds since epoch). Components not in this dict
                             use the current simulation timestamp.
            read_timestamps: Optional dict mapping component names to timestamps
                            (microseconds since epoch). Components present in this
                            dict are read with floor / sample-and-hold semantics
                            (greatest sample with `ts' <= ts`); components not in
                            this dict return the most recent sample. Past-the-end
                            timestamps clamp to the latest sample.

        Returns:
            Dict mapping read component names to their numpy array values.

        Raises:
            RuntimeError: If any component doesn't exist, has no data, or a
                          per-read timestamp is before the first sample.
            ValueError: If any write data size doesn't match the component schema

        Note:
            Write timestamps must be monotonically increasing per component. Writing
            with a timestamp less than the last write will raise an error (TimeTravel).
        """
        ...
    def truncate(self) -> None:
        """Truncate all component data and message logs in the database, resetting tick to 0.

        This clears all stored time-series data while preserving component schemas and metadata.
        The simulation tick will be reset to 0, effectively starting fresh.

        After truncate(), any subsequent write_component() calls in the same callback will write
        at the start timestamp (tick 0), preventing TimeTravel errors on the next tick.

        Use this to control the freshness of the database and ensure reliable data from a known tick.
        """
        ...
    def read_msg(
        self,
        msg_name: str,
        timestamp: Optional[int] = None,
    ) -> Optional[Any]:
        """Read a message payload from the database as a numpy uint8 array.

        Used to read sensor camera frames produced by the headless render server,
        which pushes frames to the DB at each camera's configured ``fps``.

        Args:
            msg_name: The message name (e.g., ``"drone.scene_cam"``).
            timestamp: Optional timestamp (microseconds since epoch). When ``None``
                       (the default) returns the most recent message. When provided,
                       returns the message with the greatest timestamp ``<=`` the
                       requested one (floor / sample-and-hold semantics). If the
                       requested timestamp is past the most recent message the
                       latest message is returned.

        Returns:
            NumPy ``uint8`` array containing the message payload, or ``None`` if
            no message exists at or before the requested timestamp (or no message
            exists at all).
        """
        ...
    def read_msg_at(
        self,
        msg_name: str,
        timestamp: int,
    ) -> Optional[Tuple[int, Any]]:
        """Return the selected message's timestamp and NumPy ``uint8`` payload."""
        ...
    def read_msg_latest(self, msg_name: str) -> Optional[Tuple[int, Any]]:
        """Return the latest message timestamp and NumPy ``uint8`` payload."""
        ...
    def stop_recipes(self) -> None:
        """Gracefully terminate all s10-managed recipes (external processes).

        Signals every recipe registered via ``world.recipe()`` to shut down.
        No-op if no recipes were registered or if running with ``--no-s10``.
        """
        ...

class ComponentType:
    def __init__(self, ty: PrimitiveType, shape: Tuple[int, ...]): ...
    ty: PrimitiveType
    shape: jax.typing.ArrayLike
    U64: ClassVar[ComponentType]
    F64: ClassVar[ComponentType]
    F32: ClassVar[ComponentType]
    Edge: ClassVar[ComponentType]
    Quaternion: ClassVar[ComponentType]
    SpatialPosF64: ClassVar[ComponentType]
    SpatialMotionF64: ClassVar[ComponentType]

class SystemBuilder:
    cranelift: bool
    workers: int
    def init_var(self, name: str, ty: ComponentType): ...
    def var_arrays(self) -> list[jax.typing.ArrayLike]: ...

class WorldBuilder:
    def spawn(
        self,
        archetypes: Archetype | list[Archetype],
        name: Optional[str] = None,
    ) -> EntityId: ...
    def insert(self, id: EntityId, archetypes: Archetype | Sequence[Archetype]): ...
    def sensor_camera(
        self,
        entity: EntityId,
        name: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fov: Optional[float] = None,
        near: float = 0.01,
        far: float = 1000.0,
        pos_offset: Sequence[float] = (0.0, 0.0, 0.0),
        rot_offset: Sequence[float] = (0.0, 0.0, 0.0),
        format: str = "rgba",
        effect: str = "normal",
        effect_params: Optional[dict[str, object]] = None,
        camera_model: Optional[str] = None,
        lens_hfov: Optional[float] = None,
        create_frustum: bool = False,
        show_ellipsoids: bool = False,
        frustums_color: Optional[Sequence[float]] = None,
        projection_color: Optional[Sequence[float]] = None,
        frustums_thickness: float = 0.006,
        frustums_up_marker: str = "none",
        frustums_up_marker_overlay: bool = True,
        fps: Optional[float] = None,
        cinematic: bool = False,
        ev100: Optional[float] = None,
        bloom: Optional[dict[str, object]] = None,
        environment: Optional[dict[str, object]] = None,
        frustums_face_alpha: float = 45.0 / 255.0,
        render_local: bool = False,
    ) -> None:
        """Register a virtual sensor camera on an entity.

        The headless render server emits frames into the DB at ``fps`` frames per
        second of simulation time. Frames can be read with
        ``StepContext.read_msg("entity.camera_name", timestamp=...)``.

        ``render_local=True`` instead renders the current replayed scene in a
        ``ui.sensor_view``, including in WebGL browsers. It includes scene
        particles on desktop and WebGPU, shares the schematic environment, and
        does not record video in the DB.
        ``fps`` caps this preview's wall-clock refresh rate.

        ``pos_offset`` is a body-frame translation in metres. ``rot_offset`` is
        ``[roll, pitch, yaw]`` in degrees, applied around the camera's body axes
        in intrinsic X/Y/Z order after the translation. With the default
        ``rot_offset=(0, 0, 0)``, the camera looks along body +X with body +Z up.

        Examples: forward ``(0, 0, 0)``, 15 degrees nose-down
        ``(0, -15, 0)``, 30 degrees right bank ``(30, 0, 0)``, 90 degrees left
        yaw ``(0, 0, 90)``.

        ``frustums_up_marker`` marks which frustum edge is the top of the camera
        image when ``create_frustum=True``: ``"highlight"`` thickens the
        far-plane top edge and puts a ball on the corner holding the image
        origin. ``frustums_up_marker_overlay`` repeats it along the top of this
        camera's own pane, and is on by default since that pane exists only to
        show this camera's image.

        ``cinematic=True`` enables the cinematic Earth stack in the render
        server (same meaning as KDL ``viewport cinematic=#true``). ``ev100``,
        ``bloom``, and ``environment`` match the schematic and require
        ``cinematic=True``. Omitted ``environment`` uses the house look
        (Earth, 100 klx sun, ambient 0.05). At most one cinematic owner is
        allowed: a cinematic viewport or a recorded cinematic sensor camera.
        Local previews borrow that environment and do not claim ownership of
        the atmosphere pass, which remains on the primary cinematic view.

        ``camera_model="boson640p"`` supplies 640×512, 60 Hz, the Boson+ sensor
        defaults, and an 18 degree horizontal lens. ``lens_hfov`` is converted
        to the vertical ``fov`` used by Bevy after resolving the final image
        aspect ratio. Explicit ``width``, ``height``, and ``fps`` override model
        defaults; ``fov`` and ``lens_hfov`` are mutually exclusive.

        The simulation never blocks on rendering. Pick the apparent camera
        latency at read time by reading with a timestamp offset:

            frame = ctx.read_msg("drone.scene_cam", timestamp=ctx.timestamp - 33_000)

        Raises ``ValueError`` if ``rot_offset`` is not a finite 3-element
        sequence, if ``fps`` is not a positive finite number, if
        ``frustums_up_marker`` is not a known marker, if look settings
        are passed without ``cinematic=True``, or if more than one cinematic
        environment owner is configured.
        """
        ...
    def thermal_tag(
        self,
        entity: EntityId,
        temperature_c: float,
        emissivity: float = 1.0,
    ) -> None:
        """Assign an apparent LWIR surface temperature to an entity's visual mesh."""
        ...
    def run(
        self,
        system: System,
        simulation_rate: float = 120.0,
        generate_real_time: bool = False,
        telemetry_rate: Optional[float] = None,
        default_playback_speed: float = 1.0,
        max_ticks: Optional[int] = None,
        optimize: bool = False,
        is_canceled: Optional[Callable[[], bool]] = None,
        pre_step: Optional[Callable[[int, StepContext], None]] = None,
        post_step: Optional[Callable[[int, StepContext], None]] = None,
        db_path: Optional[str] = None,
        interactive: bool = True,
        start_timestamp: Optional[int] = None,
        log_level: Optional[str] = None,
        backend: str = "cranelift",
        on_receipt: Optional[Callable[[dict], None]] = None,
        workers: int = 1,
        pre_commit: Optional[PreCommit | list[PreCommit]] = None,
        timing: Optional[Timing] = None,
        synchronous_commits: bool = False,
        determinism_trace: Optional[DeterminismTrace] = None,
    ): ...
    def prepare_component(
        self,
        name: str,
        entities: Sequence[EntityId],
        access: Literal["read", "write"] = "read",
    ) -> ComponentHandle:
        """Resolve the rows of component `name` for `entities`, in that order, for
        `pre_commit` callbacks to view. Spawning afterwards invalidates the handle."""
        ...
    def prepare_batch(self, fields: dict[str, ComponentHandle]) -> BatchHandle:
        """Group handles over the same number of rows, viewed together by
        `WorldStepContext.batch`."""
        ...
    def prepare_component_history(self, name: str) -> HistoryHandle:
        """The committed history of the `entity.component` pair `name`, for
        `WorldStepContext.read_at`."""
        ...
    def prepare_message_history(self, name: str) -> HistoryHandle:
        """The committed history of the message log `name`, for `WorldStepContext.read_at`."""
        ...
    def prepare_timestamped_writer(self, fields: dict[str, str]) -> TimestampedWriter:
        """Components that `WorldStepContext.defer_write` appends to: `fields` maps each value's
        key to an `entity.component` pair marked `external_control`."""
        ...
    def prepare_witness(
        self,
        batches: dict[str, BatchHandle],
        channels: Optional[dict[str, tuple[BatchHandle, int]]] = None,
        external: Optional[dict[str, tuple[BatchHandle, int]]] = None,
        stages: Optional[list[str]] = None,
    ) -> Witness:
        """The batches and hook channels a run's deterministic witness records. `channels` and
        `external` map each channel name to `(batch, row_bytes)`: one row per row of one of the
        batches. External channels hold what came from outside the simulation; a replay feeds
        them back, and no comparison looks at them. `stages` keeps only the stages it names."""
        ...
    def replay_pre_commit(
        self, trace: str, pre_commit: Optional[PreCommit | list[PreCommit]], live: bool = False
    ) -> dict:
        """Runs `pre_commit`, the hooks of the run that recorded `trace`, on the rows its witness
        kept exactly, without the systems or the DB, and compares what each hook writes with the
        recording. The hooks read the recorded external channels unless `live`."""
        ...
    def serve(
        self,
        system: System,
        daemon: bool = False,
        simulation_rate: float = 120.0,
        generate_real_time: bool = False,
        telemetry_rate: Optional[float] = None,
        default_playback_speed: float = 1.0,
        max_ticks: Optional[int] = None,
        addr: str = "127.0.0.1:0",
    ): ...
    def build(
        self,
        system: System,
        simulation_rate: float = 120.0,
        generate_real_time: bool = False,
        telemetry_rate: Optional[float] = None,
        default_playback_speed: float = 1.0,
        max_ticks: Optional[int] = None,
        optimize: bool = False,
        db_path: Optional[str] = None,
        backend: str = "cranelift",
        start_timestamp: Optional[int] = None,
        workers: int = 1,
        timing: Optional[Timing] = None,
    ) -> Exec: ...
    def to_jax_func(
        self,
        system: System,
        simulation_rate: float = 120.0,
        default_playback_speed: float = 1.0,
        max_ticks: Optional[int] = None,
        workers: Optional[int] = None,
    ) -> Tuple[object, list, list, object, dict, dict, dict]: ...

class EntityId:
    def __init__(self, id: int): ...

class SpatialTransform:
    __metadata__: ClassVar[Tuple[Component,]]
    def __init__(
        self,
        arr: Optional[jax.typing.ArrayLike] = None,
        angular: Optional[Quaternion] = None,
        linear: Optional[jax.typing.ArrayLike] = None,
    ): ...
    def flatten(self) -> Any: ...
    @staticmethod
    def unflatten(aux: Any, jax: Any) -> Any: ...
    @staticmethod
    def from_array(arr: jax.typing.ArrayLike) -> SpatialTransform: ...
    def linear(self) -> jax.Array: ...
    def angular(self) -> Quaternion: ...
    def asarray(self) -> jax.typing.ArrayLike: ...
    @overload
    def __add__(self, other: SpatialTransform) -> SpatialTransform: ...
    @overload
    def __add__(self, other: SpatialMotion) -> SpatialTransform: ...

class SpatialForce:
    __metadata__: ClassVar[Tuple[Component,]]
    def __init__(
        self,
        arr: Optional[jax.typing.ArrayLike] = None,
        torque: Optional[jax.typing.ArrayLike] = None,
        linear: Optional[jax.typing.ArrayLike] = None,
    ): ...
    @staticmethod
    def from_array(arr: jax.typing.ArrayLike) -> SpatialForce: ...
    def flatten(self) -> Any: ...
    @staticmethod
    def unflatten(aux: Any, jax: Any) -> Any: ...
    def asarray(self) -> jax.typing.ArrayLike: ...
    def force(self) -> jax.typing.ArrayLike: ...
    def torque(self) -> jax.typing.ArrayLike: ...
    def __add__(self, other: SpatialForce) -> SpatialForce: ...

class SpatialMotion:
    __metadata__: ClassVar[Tuple[Component,]]
    def __init__(
        self,
        angular: Optional[jax.typing.ArrayLike] = None,
        linear: Optional[jax.typing.ArrayLike] = None,
    ): ...
    @staticmethod
    def from_array(arr: jax.typing.ArrayLike) -> SpatialMotion: ...
    def flatten(self) -> Any: ...
    @staticmethod
    def unflatten(aux: Any, jax: Any) -> Any: ...
    def asarray(self) -> jax.typing.ArrayLike: ...
    def linear(self) -> jax.Array: ...
    def angular(self) -> jax.Array: ...
    def __add__(self, other: SpatialMotion) -> SpatialMotion: ...

class SpatialInertia:
    # Sign convention: `inertia_off_diag` holds the signed off-diagonal entries
    # of the inertia tensor matrix ([Ixy, Ixz, Iyz], Ixy = -integral(x*y dm)),
    # not products of inertia. Negate CAD products of inertia before passing.
    __metadata__: ClassVar[Tuple[Component,]]
    def __init__(
        self,
        mass: jax.typing.ArrayLike,
        inertia: Optional[jax.typing.ArrayLike] = None,
        inertia_off_diag: Optional[jax.typing.ArrayLike] = None,
    ): ...
    @staticmethod
    def from_array(arr: jax.typing.ArrayLike) -> SpatialInertia: ...
    def flatten(self) -> Any: ...
    @staticmethod
    def unflatten(aux: Any, jax: Any) -> Any: ...
    def asarray(self) -> jax.typing.ArrayLike: ...
    def mass(self) -> jax.typing.ArrayLike: ...
    def inertia_diag(self) -> jax.typing.ArrayLike: ...
    def inertia_off_diag(self) -> jax.typing.ArrayLike: ...
    def inertia_matrix(self) -> jax.Array: ...

class Quaternion:
    __metadata__: ClassVar[Tuple[Component,]]
    def __init__(self, arr: jax.typing.ArrayLike): ...
    @staticmethod
    def from_array(arr: jax.typing.ArrayLike) -> Quaternion: ...
    def flatten(self) -> Any: ...
    @staticmethod
    def unflatten(aux: Any, jax: Any) -> Any: ...
    def asarray(self) -> jax.typing.ArrayLike: ...
    @staticmethod
    def identity() -> Quaternion: ...
    @staticmethod
    def from_axis_angle(axis: jax.typing.ArrayLike, angle: jax.typing.ArrayLike) -> Quaternion: ...
    def vector(self) -> jax.Array: ...
    def normalize(self) -> Quaternion: ...
    def __mul__(self, other: Quaternion) -> Quaternion: ...
    def __add__(self, other: Quaternion) -> Quaternion: ...
    @overload
    def __matmul__(self, vector: jax.Array) -> jax.Array: ...
    @overload
    def __matmul__(self, spatial_transform: SpatialTransform) -> SpatialTransform: ...
    @overload
    def __matmul__(self, spatial_motion: SpatialMotion) -> SpatialMotion: ...
    @overload
    def __matmul__(self, spatial_force: SpatialForce) -> SpatialForce: ...
    def inverse(self) -> Quaternion: ...
    def integrate_body(self, body_delta: jax.Array) -> Quaternion: ...

class QueryInner:
    def join_query(self, other: QueryInner) -> QueryInner: ...
    def arrays(self) -> list[jax.Array]: ...
    def map(self, ty: jax.Array, f: Component) -> Any: ...
    @staticmethod
    def from_builder(
        sys: SystemBuilder, component_ids: list[str], args: list[Any]
    ) -> QueryInner: ...
    def output(self, builder: SystemBuilder, args: list[Any]) -> Any: ...

class GraphQueryInner:
    def arrays(
        self, from_query: QueryInner, to_query: QueryInner
    ) -> dict[int, Tuple[list[jax.Array], list[jax.Array]]]: ...
    @staticmethod
    def from_builder(builder: SystemBuilder, edge_name: str, reverse: bool) -> GraphQueryInner: ...
    @staticmethod
    def from_builder_total_edge(builder: SystemBuilder) -> GraphQueryInner: ...
    def map(
        self,
        from_query: QueryInner,
        to_query: QueryInner,
        ty: jax.typing.ArrayLike,
        f: Component,
    ) -> QueryInner: ...

class Edge:
    __metadata__: ClassVar[Tuple[Component,]]
    def __init__(self, left: EntityId, right: EntityId): ...
    def flatten(self) -> Any: ...
    @staticmethod
    def unflatten(aux: Any, jax: Any) -> Any: ...

class Recording:
    """When a component's values reach the DB, counted in commits."""

    @staticmethod
    def on_change() -> Recording:
        """When its bytes differ from the last recorded value; the default."""
        ...
    @staticmethod
    def every_commit() -> Recording:
        """At every commit, changed or not."""
        ...
    @staticmethod
    def every(n: int) -> Recording:
        """At every `n`th commit, changed or not."""
        ...
    @staticmethod
    def manual() -> Recording:
        """At the commits a `pre_commit` callback asks for with `WorldStepContext.record`."""
        ...
    @staticmethod
    def initial_only() -> Recording:
        """Once, when the run starts."""
        ...
    @staticmethod
    def transient() -> Recording:
        """Never: the component stays out of the DB, its history, replication and the editor."""
        ...

class Component:
    asset: bool
    def __init__(
        self,
        name: str,
        ty: Optional[ComponentType] = None,
        asset: bool = False,
        metadata: dict[str, str | bool | int] = {},
        recording: Optional[Recording] = None,
    ): ...
    @staticmethod
    def id(component: Any) -> str:
        """
        DEPRECATED: Use `Component.name()` instead.
        """
    @staticmethod
    def name(component: Any) -> str: ...
    @staticmethod
    def index(component: Any) -> ShapeIndexer: ...
    @staticmethod
    def of(component: Any) -> Component: ...

class ShapeIndexer:
    def __getitem__(self, index: Any) -> ShapeIndexer: ...

class Exec:
    def run(
        self,
        ticks: int = 1,
        show_progress: bool = True,
        is_canceled: Optional[Callable[[], bool]] = None,
    ): ...
    def last_receipt(self) -> Optional[dict]:
        """Phase timings, DB sync counters and backend facts of the latest `run`."""
        ...
    def profile(self) -> dict[str, float]: ...
    def save_archive(self, path: str, format: str): ...
    def history(self, components: str | list[str]) -> pl.DataFrame: ...

class S10RestartPolicy:
    Never: ClassVar[S10RestartPolicy]
    Instant: ClassVar[S10RestartPolicy]

class S10Ready:
    @staticmethod
    def tcp(addr: str) -> S10Ready: ...
    @staticmethod
    def unix(path: str) -> S10Ready: ...
    @staticmethod
    def file(path: str) -> S10Ready: ...
    @staticmethod
    def log(pattern: str) -> S10Ready: ...
    @staticmethod
    def delay(ms: int) -> S10Ready: ...

class S10PyRecipe:
    def __init__(
        self,
        name: str,
        path: str | None = None,
        addr: str | None = None,
        optimize: bool | None = None,
        env: dict[str, str] | None = None,
        depends_on: list[str] | None = None,
        ready: S10Ready | None = None,
        ready_timeout: str | None = None,
    ): ...
    @staticmethod
    def cargo(
        name: str,
        path: str,
        package: str | None = None,
        bin: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        restart_policy: S10RestartPolicy | None = None,
        depends_on: list[str] | None = None,
        ready: S10Ready | None = None,
        ready_timeout: str | None = None,
        silence: bool = False,
    ) -> S10PyRecipe: ...
    @staticmethod
    def process(
        name: str,
        cmd: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        restart_policy: S10RestartPolicy | None = None,
        depends_on: list[str] | None = None,
        ready: S10Ready | None = None,
        ready_timeout: str | None = None,
        silence: bool = False,
    ) -> S10PyRecipe: ...
    def to_json(self) -> str: ...
    def name(self) -> str: ...

class _S10Module:
    PyRecipe: ClassVar[type[S10PyRecipe]]
    RestartPolicy: ClassVar[type[S10RestartPolicy]]
    Ready: ClassVar[type[S10Ready]]

s10: _S10Module

class GraphEntity:
    def __init__(self, entity_id: EntityId, *components: ShapeIndexer | Any): ...

def six_dof(
    time_step: float | None = None,
    sys: Any = None,
    integrator: Integrator = Integrator.Rk4,
) -> System: ...
def skew(arr: jax.Array) -> jax.Array: ...

class System:
    def pipe(self, other: System) -> System: ...
    def __or__(self, other: System | None) -> System: ...
    def __ror__(self, other: Any) -> System: ...

class PyFnSystem:
    def __init__(
        self,
        sys: Any,
        input_ids: list[str],
        output_ids: list[str],
        edge_ids: list[str],
        name: str,
    ): ...
    def system(self) -> System: ...

class MonteCarloParam:
    def __init__(
        self,
        type_: Any,
        default: Any = None,
        min: Any = None,
        max: Any = None,
    ): ...

class MonteCarloParamsSpec:
    def to_json(self) -> str: ...

class MonteCarloParams:
    run_id: str | None
    seed: int | None
    db_path: str | None
    db_addr: str | None
    cache_dir: str | None
    run_dir: str | None
    meta: dict[str, Any]
    def get(self, key: str, default: Any = None) -> Any: ...
    def __getitem__(self, key: str) -> Any: ...
    def as_overrides_dict(self) -> dict[str, Any]: ...
    def slots(self) -> dict[str, Any]: ...
    def ports(self) -> dict[str, int]: ...
