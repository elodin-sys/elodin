from __future__ import annotations
from collections.abc import Sequence
import jax
from typing import (
    Any,
    Optional,
    Union,
    Tuple,
    ClassVar,
    List,
    Protocol,
    Annotated,
    overload,
)
import polars as pl
from elodin import Archetype

class PrimitiveType:
    F64: PrimitiveType
    U64: PrimitiveType

class Integrator:
    Rk4: Integrator
    SemiImplicit: Integrator

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

class Asset(Protocol):
    def asset_name(self) -> str: ...
    def bytes(self) -> bytes: ...

class WorldBuilder:
    def spawn(
        self,
        archetypes: Asset | Archetype | list[Archetype],
        name: Optional[str] = None,
    ) -> EntityId: ...
    def insert(
        self, id: EntityId, archetypes: Asset | Archetype | Sequence[Archetype]
    ): ...
    def insert_asset(self, asset: Asset) -> Handle: ...
    def run(
        self,
        system: Any,
        sim_time_step: Optional[float] = None,
        run_time_step: Optional[float] = None,
        output_time_step: Optional[float] = None,
        max_ticks: Optional[int] = None,
        client: Optional[Client] = None,
    ): ...
    def serve(
        self,
        system: Any,
        daemon: bool = False,
        sim_time_step: Optional[float] = None,
        run_time_step: Optional[float] = None,
        output_time_step: Optional[float] = None,
        max_ticks: Optional[int] = None,
        client: Optional[Client] = None,
        addr: Optional[str] = None,
    ): ...
    def build(
        self,
        system: Any,
        sim_time_step: Optional[float] = None,
        run_time_step: Optional[float] = None,
        output_time_step: Optional[float] = None,
        client: Optional[Client] = None,
    ) -> Exec: ...

class EntityId:
    def __init__(self, id: int): ...

class Client:
    @staticmethod
    def cpu() -> Client: ...

class SpatialTransform:
    __metadata__: ClassVar[Tuple[Component,]]
    def __init__(
        self,
        arr: Optional[jax.typing.ArrayLike] = None,
        angular: Optional[Quaternion] = None,
        linear: Optional[jax.typing.ArrayLike] = None,
    ): ...
    @staticmethod
    def from_linear(linear: jax.typing.ArrayLike) -> SpatialTransform:
        """
        DEPRECATED: Use `SpatialTransform(linear=...)` instead.
        """
    @staticmethod
    def from_angular(
        quaternion: jax.typing.ArrayLike | Quaternion,
    ) -> SpatialTransform:
        """
        DEPRECATED: Use `SpatialTransform(angular=...)` instead.
        """
    @staticmethod
    def from_axis_angle(
        axis: jax.typing.ArrayLike, angle: jax.typing.ArrayLike
    ) -> SpatialTransform:
        """
        DEPRECATED: Use `SpatialTransform(angular=Quaternion.from_axis_angle(...))` instead.
        """
    def flatten(self) -> Any: ...
    @staticmethod
    def unflatten(aux: Any, jax: Any) -> Any: ...
    @staticmethod
    def from_array(arr: jax.typing.ArrayLike) -> SpatialTransform: ...
    @staticmethod
    def zero() -> SpatialTransform:
        """
        DEPRECATED: Use `SpatialTransform()` instead.
        """
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
    @staticmethod
    def zero() -> SpatialForce:
        """
        DEPRECATED: Use `SpatialForce()` instead.
        """
    @staticmethod
    def from_linear(linear: jax.typing.ArrayLike) -> SpatialForce:
        """
        DEPRECATED: Use `SpatialForce(linear=...)` instead.
        """
    @staticmethod
    def from_torque(torque: jax.typing.ArrayLike) -> SpatialForce:
        """
        DEPRECATED: Use `SpatialForce(torque=...)` instead.
        """
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
    @staticmethod
    def zero() -> SpatialMotion: ...
    @staticmethod
    def from_linear(linear: jax.typing.ArrayLike) -> SpatialMotion:
        """
        DEPRECATED: Use `SpatialMotion(linear=...)` instead.
        """
    @staticmethod
    def from_angular(angular: jax.typing.ArrayLike) -> SpatialMotion:
        """
        DEPRECATED: Use `SpatialMotion(angular=...)` instead.
        """
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
    def from_axis_angle(
        axis: jax.typing.ArrayLike, angle: jax.typing.ArrayLike
    ) -> Quaternion: ...
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

class Mesh:
    @staticmethod
    def cuboid(x: float, y: float, z: float) -> Mesh: ...
    @staticmethod
    def sphere(radius: float) -> Mesh: ...
    def asset_name(self) -> str: ...
    def bytes(self) -> bytes: ...

class Material:
    @staticmethod
    def color(r: float, g: float, b: float) -> Material: ...
    def asset_name(self) -> str: ...
    def bytes(self) -> bytes: ...

class Texture: ...

class Handle:
    __metadata__: ClassVar[Tuple[Component,]]
    def flatten(self) -> Any: ...
    @staticmethod
    def unflatten(aux: Any, jax: Any) -> Any: ...

class Pbr:
    def __init__(self, mesh: Mesh, material: Material): ...
    @staticmethod
    def from_url(url: str) -> Pbr: ...
    @staticmethod
    def from_path(path: str) -> Pbr: ...
    def asset_name(self) -> str: ...
    def bytes(self) -> bytes: ...

class Metadata:
    ty: ComponentType
    @staticmethod
    def of(component: Annotated[Any, Component]) -> Metadata: ...

class QueryInner:
    def join_query(self, other: QueryInner) -> QueryInner: ...
    def arrays(self) -> list[jax.Array]: ...
    def map(self, ty: jax.Array, f: Metadata) -> Any: ...
    @staticmethod
    def from_builder(
        sys: SystemBuilder, componnet_ids: list[str], args: list[any]
    ) -> QueryInner: ...
    def output(self, builder: SystemBuilder, args: list[Any]) -> Any: ...

class GraphQueryInner:
    def arrays(
        self, from_query: QueryInner, to_query: QueryInner
    ) -> dict[int, Tuple[list[jax.Array], list[jax.Array]]]: ...
    @staticmethod
    def from_builder(
        builder: SystemBuilder, edge_name: str, reverse: bool
    ) -> GraphQueryInner: ...
    @staticmethod
    def from_builder_total_edge(builder: SystemBuilder) -> GraphQueryInner: ...
    def map(
        self,
        from_query: QueryInner,
        to_query: QueryInner,
        ty: jax.typing.ArrayLike,
        f: Metadata,
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

class ShapeIndexer:
    def __getitem__(self, index: Any) -> ShapeIndexer: ...

class Impeller:
    @staticmethod
    def tcp(addr: str) -> Impeller: ...

class Exec:
    def run(self, ticks: int = 1, show_progress: bool = True): ...
    def profile(self) -> dict[str, float]: ...
    def write_to_dir(self, path: str): ...
    def history(self) -> pl.DataFrame: ...
    def column_array(self, name: str) -> pl.Series: ...

class Color:
    def __init__(self, r: float, g: float, b: float): ...
    TURQUOISE: ClassVar[Color]
    SLATE: ClassVar[Color]
    PUMPKIN: ClassVar[Color]
    YOLK: ClassVar[Color]
    PEACH: ClassVar[Color]
    REDDISH: ClassVar[Color]
    HYPERBLUE: ClassVar[Color]
    MINT: ClassVar[Color]

class Gizmo:
    @staticmethod
    def vector(name: str, offset: int, color: Color) -> jax.Array: ...

class Panel:
    @staticmethod
    def vsplit(*panels: Panel, active: bool = False) -> Panel: ...
    @staticmethod
    def hsplit(*panels: Panel, active: bool = False) -> Panel: ...
    @staticmethod
    def viewport(
        track_entity: Optional[EntityId] = None,
        track_rotation: bool = True,
        fov: float = 45.0,
        active: bool = False,
        pos: Union[List[float], jax.Array, None] = None,
        looking_at: Union[List[float], jax.Array, None] = None,
        show_grid: bool = False,
        hdr: bool = False,
        name: Optional[str] = None,
    ) -> Panel: ...
    @staticmethod
    def graph(*entities: GraphEntity, name: Optional[str] = None) -> Panel: ...
    def asset_name(self) -> str: ...
    def bytes(self) -> bytes: ...

class GraphEntity:
    def __init__(self, entity_id: EntityId, *components: ShapeIndexer | Any): ...

class Glb:
    def __init__(self, path: str): ...
    def bytes(self) -> bytes: ...

class BodyAxes:
    def __init__(self, entity: EntityId, scale: float = 1.0): ...
    def asset_name(self) -> str: ...
    def bytes(self) -> bytes: ...

class VectorArrow:
    def __init__(
        self,
        entity: EntityId,
        component_name: str,
        offset: int = 0,
        color: Optional[Color] = None,
        attached: bool = False,
        body_frame: bool = True,
        scale: float = 1.0,
    ): ...
    def asset_name(self) -> str: ...
    def bytes(self) -> bytes: ...

class Line3d:
    def __init__(
        self,
        entity: EntityId,
        component_name: str = "world_pos",
        line_width: float = 10.0,
        color: Optional[Color] = None,
        index: Optional[list[int]] = None,
        perspective: bool = False,
    ): ...
    def asset_name(self) -> str: ...
    def bytes(self) -> bytes: ...

def six_dof(
    time_step: float | None = None,
    sys: Any = None,
    integrator: Integrator = Integrator.Rk4,
) -> System: ...
def read_batch_results(path: str) -> Tuple[list[pl.DataFrame], list[int]]: ...
def skew(arr: jax.Array) -> jax.Array: ...

class System:
    def pipe(self, other: System) -> System: ...
    def __or__(self, other: System) -> System: ...

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
