from .elodin import *
from typing import Protocol, Generic, TypeVar, Any, Callable, Annotated, Type, Union
from typing_extensions import TypeVarTuple, Unpack
from dataclasses import dataclass
from jax.tree_util import tree_flatten, tree_unflatten
import inspect
import jax
import typing
import numpy

__doc__ = elodin.__doc__

jax.config.update("jax_enable_x64", True)

Self = TypeVar("Self")


class System(Protocol):
    @staticmethod
    def call(builder: PipelineBuilder):
        ...

    def init(builder: PipelineBuilder) -> PipelineBuilder:
        ...

    def pipe(self, other: Self) -> Self:
        return Pipe(self, other)


@dataclass
class Pipe(System):
    a: System
    b: System

    def init(self, builder):
        self.a.init(builder)
        self.b.init(builder)

    def call(self, builder):
        self.a.call(builder)
        self.b.call(builder)


def system(func) -> Callable[[Any], None]:
    class Inner(System):
        func

        def init(self, builder):
            sig = inspect.signature(func)
            params = sig.parameters
            for _, p in params.items():
                p.annotation.init_builder(p.annotation, builder)
            if sig.return_annotation is not inspect._empty:
                sig.return_annotation.init_builder(sig.return_annotation, builder)

        def call(self, builder):
            sig = inspect.signature(func)
            params = sig.parameters
            args = [
                p.annotation.from_builder(p.annotation, builder)
                for (_, p) in params.items()
            ]
            ret = func(*args)
            if ret is not None:
                ret.insert_into_builder(builder)

    inner = Inner()
    inner.func = func
    return inner


@dataclass
class ComponentData:
    id: ComponentId
    type: ComponentType
    asset: bool
    from_expr: Callable[[Any], Any]
    name: str
    def to_metadata(self) -> Metadata:
        metadata = Metadata(self.id, self.type, self.name)
        return metadata


O = TypeVar("O")
T = TypeVar("T", bound="Union[jax.Array, FromArray]")
Q = TypeVar("Q", bound="Query[Any]")


A = TypeVarTuple("A")
B = TypeVar("B", bound="Query[Any]")


class Query(Generic[Unpack[A]]):
    bufs: list[jax.Array]
    component_data: list[ComponentData]
    inner: QueryInner

    def __init__(self, inner: QueryInner, component_data: list[ComponentData]):
        self.bufs = inner.arrays()
        self.inner = inner
        self.component_data = component_data

    def map(self, out_tp: type[O], f: Callable[[*A], O]) -> Q:
        buf = jax.vmap(
            lambda b: f(
                *[data.from_expr(x) for (x, data) in zip(b, self.component_data)]
            ),
            in_axes=0,
            out_axes=0,
        )(self.bufs)
        (bufs, _) = tree_flatten(buf)
        component_data = out_tp.__metadata__[0]
        return Query(self.inner.map(bufs[0], component_data.to_metadata()), [component_data])

    def join(self, other) -> B:
        return Query(self.inner.join_query(other.inner), self.component_data + other.component_data)

    @staticmethod
    def from_builder(new_tp: type[B], builder: PipelineBuilder) -> Q:
        t_args = typing.get_args(new_tp)
        ids = []
        component_data = []
        for t_arg in t_args:
            data = t_arg.__metadata__[0]
            component_data.append(data)
            ids.append(data.id)
        return Query(QueryInner.from_builder(builder, ids), component_data)

    @staticmethod
    def init_builder(new_tp: Type[B], builder: PipelineBuilder):
        t_args = typing.get_args(new_tp)
        for t_arg in t_args:
            component_data: ComponentData = t_arg.__metadata__[0]
            buf = builder.init_var(component_data.id, component_data.type)

    def insert_into_builder(self, builder: PipelineBuilder):
        self.inner.insert_into_builder(builder)


E = TypeVar("E")
class GraphQuery(Generic[E, Unpack[A]]):
    bufs: dict[int, (list[jax.Array], list[jax.Array])]
    component_data: list[ComponentData]
    inner: GraphQueryInner
    def __init__(self, inner: GraphQueryInner, component_data: list[ComponentData]):
        self.bufs = inner.arrays()
        self.inner = inner
        self.component_data = component_data

    @staticmethod
    def from_builder(new_tp: type[B], builder: PipelineBuilder) -> Q:
        t_args = typing.get_args(new_tp)
        ids = []
        component_data = []
        edge_ty = t_args[0]
        edge_id = edge_ty.__metadata__[0].id
        for t_arg in t_args[1:]:
            data = t_arg.__metadata__[0]
            component_data.append(data)
            ids.append(data.id)
        return GraphQuery(GraphQueryInner.from_builder(builder, edge_id, ids), component_data)
    @staticmethod
    def init_builder(new_tp: Type[B], builder: PipelineBuilder):
        t_args = typing.get_args(new_tp)
        for t_arg in t_args:
            component_data: ComponentData = t_arg.__metadata__[0]
            buf = builder.init_var(component_data.id, component_data.type)
    def edge_fold(self, out_tp: type[O], init_value: O, fn: Callable[[O, A, A], O]) -> Q:
        out_bufs = []
        queries = []
        init_value_flat, init_value_tree = tree_flatten(init_value)
        for (i, (f, to)) in self.bufs.items():
            def vmap_inner(a):
                (f, to) = a
                def scan_inner(xs, to):
                    xs = tree_unflatten(init_value_tree, xs)
                    args = [data.from_expr(x) for (x, data) in zip(f, self.component_data)] + [data.from_expr(x) for (x, data) in zip(to, self.component_data)]
                    o = fn(xs, *args)
                    o_flat,_ = tree_flatten(o)
                    return (o_flat, 0)
                scan_out = jax.lax.scan(scan_inner, init_value_flat, to)[0]
                return scan_out

            buf = jax.vmap(vmap_inner)((f, to))
            (new_bufs, _) = tree_flatten(buf)
            if len(out_bufs) == 0:
                out_bufs = new_bufs
            else:
                out_bufs = [jax.numpy.concatenate([x, y]) for (x, y) in zip(out_bufs, new_bufs)]
            component_data = out_tp.__metadata__[0]
        return Query(self.inner.map(out_bufs[0], component_data.to_metadata()), [component_data])


class SystemParam(Protocol):
    @staticmethod
    def from_builder(builder: PipelineBuilder) -> Any:
        ...


class FromArray(Protocol):
    @staticmethod
    def from_array(arr: jax.Array) -> Any:
        ...


class Component:
    def __class_getitem__(cls, params):
        def parse_id(id):
            if isinstance(id, str) or isinstance(id, int):
                return ComponentId(id)
            else:
                return id

        def from_expr(ty):
            if ty is jax.Array:
                return lambda x: x
            else:
                return ty.from_array

        if len(params) == 4:
            (t, raw_id, type, asset) = params
            id = parse_id(raw_id)
            return Annotated.__class_getitem__(
                (t, ComponentData(id, type, asset, from_expr(t), f"{raw_id}"))
            )  # type: ignore
        if len(params) == 3:
            (t, raw_id, type) = params
            id = parse_id(raw_id)
            return Annotated.__class_getitem__(
                (t, ComponentData(id, type, False, from_expr(t), f"{raw_id}"))
            )  # type: ignore
        elif len(params) == 2:
            (t, raw_id) =params
            id = parse_id(raw_id)
            type = t.__metadata__[0].type
            return Annotated.__class_getitem__(
                (t, ComponentData(id, type, False, from_expr(t), f"{raw_id}"))
            )  # type: ignore
        else:
            raise Exception("Component must be called an ID and type")


class Archetype(Protocol):
    def archetype_id(self) -> int:
        return abs(hash(type(self).__name__))

    def component_data(self) -> list[ComponentData]:
        return [
            v.__metadata__[0]
            for v in typing.get_type_hints(self, include_extras=True).values()
        ]

    def arrays(self) -> list[jax.Array]:
        return [
            numpy.asarray(tree_flatten(v)[0][0])
            for (a, v) in self.__dict__.items()
            if not a.startswith("__") and not callable(getattr(self, a))
        ]


jax.tree_util.register_pytree_node(
    elodin.SpatialTransform,
    elodin.SpatialTransform.flatten,
    elodin.SpatialTransform.unflatten,
)
jax.tree_util.register_pytree_node(
    elodin.SpatialMotion, elodin.SpatialMotion.flatten, elodin.SpatialMotion.unflatten
)
jax.tree_util.register_pytree_node(
    elodin.SpatialForce, elodin.SpatialForce.flatten, elodin.SpatialForce.unflatten
)
jax.tree_util.register_pytree_node(
    elodin.SpatialInertia,
    elodin.SpatialInertia.flatten,
    elodin.SpatialInertia.unflatten,
)
jax.tree_util.register_pytree_node(
    elodin.Quaternion, elodin.Quaternion.flatten, elodin.Quaternion.unflatten
)
jax.tree_util.register_pytree_node(
    elodin.Handle, elodin.Handle.flatten, elodin.Handle.unflatten
)
jax.tree_util.register_pytree_node(
    elodin.Edge, elodin.Edge.flatten, elodin.Edge.unflatten
)

WorldPos = Component[SpatialTransform, "world_pos", ComponentType.SpatialPosF64]
WorldVel = Component[SpatialMotion, "world_vel", ComponentType.SpatialMotionF64]
WorldAccel = Component[SpatialMotion, "world_accel", ComponentType.SpatialMotionF64]
Force = Component[SpatialForce, "force", ComponentType.SpatialMotionF64]
Inertia = Component[SpatialInertia, "inertia", ComponentType.SpatialPosF64]
PbrAsset = Component[Handle, 2241, ComponentType.U64, True]
EntityMetadataAsset = Component[Handle, 2242, ComponentType.U64, True]


@dataclass
class Body(Archetype):
    world_pos: WorldPos = WorldPos.zero()
    world_vel: WorldVel = WorldVel.zero()
    inertia: Inertia = Inertia.from_mass(1.0)
    pbr: PbrAsset = Pbr(Mesh.sphere(1.0), Material.color(1.0, 1.0, 1.0))
    force: Force = Force.zero()
    world_accel: WorldAccel = WorldAccel.zero()


def build_expr(builder: PipelineBuilder, sys: System) -> Any:
    sys.init(builder)

    def call(args, builder):
        builder.inject_args(args)
        sys.call(builder)

    xla = jax.xla_computation(lambda a: call(a, builder))(builder.var_arrays())
    return xla
