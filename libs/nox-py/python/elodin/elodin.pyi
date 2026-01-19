from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import (
    Any,
    ClassVar,
    Optional,
    Tuple,
    overload,
)

import jax
import polars as pl

from elodin import Archetype

class PrimitiveType:
    F64: PrimitiveType
    U64: PrimitiveType

class Integrator:
    Rk4: Integrator
    SemiImplicit: Integrator

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
    def read_component(self, pair_name: str) -> jax.Array:
        """Read the latest component data from the database.

        Args:
            pair_name: The full component name in "entity.component" format
                      (e.g., "drone.accel", "drone.gyro", "drone.world_pos")

        Returns:
            NumPy array containing the component data (dtype matches component schema).
            The array is always 1D; reshape if needed.

        Raises:
            RuntimeError: If the component doesn't exist or has no data
        """
        ...
    def component_batch_operation(
        self,
        reads: list[str] = [],
        writes: Optional[dict[str, jax.typing.ArrayLike]] = None,
        write_timestamps: Optional[dict[str, int]] = None,
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

        Returns:
            Dict mapping read component names to their numpy array values.

        Raises:
            RuntimeError: If any component doesn't exist or has no data
            ValueError: If any write data size doesn't match the component schema

        Note:
            Timestamps must be monotonically increasing per component. Writing with
            a timestamp less than the last write will raise an error (TimeTravel).
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
    def init_var(self, name: str, ty: ComponentType): ...
    def var_arrays(self) -> list[jax.typing.ArrayLike]: ...

class WorldBuilder:
    def spawn(
        self,
        archetypes: Archetype | list[Archetype],
        name: Optional[str] = None,
    ) -> EntityId: ...
    def insert(self, id: EntityId, archetypes: Archetype | Sequence[Archetype]): ...
    def run(
        self,
        system: System,
        sim_time_step: float = 1 / 120.0,
        run_time_step: Optional[float] = None,
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
    ): ...
    def serve(
        self,
        system: System,
        daemon: bool = False,
        sim_time_step: float = 1 / 120.0,
        run_time_step: Optional[float] = None,
        default_playback_speed: float = 1.0,
        max_ticks: Optional[int] = None,
        addr: str = "127.0.0.1:0",
    ): ...
    def build(
        self,
        system: System,
        sim_time_step: float = 1 / 120.0,
        run_time_step: Optional[float] = None,
        default_playback_speed: float = 1.0,
        optimize: bool = False,
    ) -> Exec: ...
    def to_jax_func(
        self,
        system: System,
        sim_time_step: float = 1 / 120.0,
        run_time_step: Optional[float] = None,
        default_playback_speed: float = 1.0,
        max_ticks: Optional[int] = None,
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
    __metadata__: ClassVar[Tuple[Component,]]
    def __init__(
        self, mass: jax.typing.ArrayLike, inertia: Optional[jax.typing.ArrayLike] = None
    ): ...
    @staticmethod
    def from_array(arr: jax.typing.ArrayLike) -> SpatialInertia: ...
    def flatten(self) -> Any: ...
    @staticmethod
    def unflatten(aux: Any, jax: Any) -> Any: ...
    def asarray(self) -> jax.typing.ArrayLike: ...
    def mass(self) -> jax.typing.ArrayLike: ...
    def inertia_diag(self) -> jax.typing.ArrayLike: ...

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

class Component:
    asset: bool
    def __init__(
        self,
        name: str,
        ty: Optional[ComponentType] = None,
        asset: bool = False,
        metadata: dict[str, str | bool | int] = {},
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

class Impeller:
    @staticmethod
    def tcp(addr: str) -> Impeller: ...

class Exec:
    def run(
        self,
        ticks: int = 1,
        show_progress: bool = True,
        is_canceled: Optional[Callable[[], bool]] = None,
    ): ...
    def profile(self) -> dict[str, float]: ...
    def save_archive(self, path: str, format: str): ...
    def history(self, components: str | list[str]) -> pl.DataFrame: ...

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
