from .elodin import *
from typing import (
    Protocol,
    Generic,
    TypeVar,
    Any,
    Callable,
    Annotated,
    Type,
    Union,
    Optional,
    Tuple,
)
from typing_extensions import TypeVarTuple, Unpack
from dataclasses import dataclass
from jax.tree_util import tree_flatten, tree_unflatten
import inspect
import jax
import typing
import numpy
import code
import readline
import rlcompleter
import re


__doc__ = elodin.__doc__  # type: ignore

jax.config.update("jax_enable_x64", True)

Self = TypeVar("Self")


class System(Protocol):
    @staticmethod
    def call(builder: PipelineBuilder): ...

    def init(self, builder: PipelineBuilder) -> PipelineBuilder: ...

    def pipe(self, other: Any) -> Any:
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


def system(func) -> System:
    class Inner(System):
        func: Callable[[Any], Any]

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


O = TypeVar("O")
T = TypeVar("T", bound="Union[jax.Array, FromArray]")
Q = TypeVar("Q", bound="Query[Any]")


A = TypeVarTuple("A")


class Query(Generic[Unpack[A]]):
    bufs: list[jax.Array]
    component_data: list[Component]
    component_classes: list[type[Any]]
    inner: QueryInner

    def __init__(
        self,
        inner: QueryInner,
        component_data: list[Component],
        component_classes: list[type[Any]],
    ):
        self.bufs = inner.arrays()
        self.inner = inner
        self.component_data = component_data
        self.component_classes = component_classes

    def map(self, out_tp: type[O], f: Callable[[*A], O]) -> "Query[O]":
        buf = jax.vmap(
            lambda b: f(
                *[from_array(cls, x) for (x, cls) in zip(b, self.component_classes)]
            ),
            in_axes=0,
            out_axes=0,
        )(self.bufs)
        (bufs, _) = tree_flatten(buf)
        component_data = out_tp.__metadata__[0]  # type: ignore
        return Query(
            self.inner.map(bufs[0], component_data.to_metadata()),
            [component_data],
            [out_tp],
        )

    def join(self, other: "Query[Unpack[A]]") -> "Query[Any]":
        return Query(
            self.inner.join_query(other.inner),
            self.component_data + other.component_data,
            self.component_classes + other.component_classes,
        )

    @staticmethod
    def from_builder(new_tp: type[Any], builder: PipelineBuilder) -> "Query[Any]":
        t_args = typing.get_args(new_tp)
        ids = []
        component_data = []
        component_classes = []
        for t_arg in t_args:
            data = t_arg.__metadata__[0]
            component_data.append(data)
            component_classes.append(t_arg)
            ids.append(Component.id(t_arg))
        return Query(
            QueryInner.from_builder(builder, ids), component_data, component_classes
        )

    @staticmethod
    def init_builder(new_tp: type[Any], builder: PipelineBuilder):
        t_args = typing.get_args(new_tp)
        for t_arg in t_args:
            component_data: Component = t_arg.__metadata__[0]
            buf = builder.init_var(Component.id(t_arg), component_data.ty)

    def __getitem__(self, index: int) -> Any:
        if len(self.bufs) > 1:
            raise Exception("Cannot index into a query with multiple inputs")
        cls = self.component_classes[0]
        return from_array(cls, self.bufs[0][index])

    def insert_into_builder(self, builder: PipelineBuilder):
        self.inner.insert_into_builder(builder)


def from_array(cls, arr):
    if hasattr(cls, "__origin__"):
        cls = cls.__origin__
    if cls is jax.Array:
        return arr
    else:
        return cls.from_array(arr)


E = TypeVar("E")


class GraphQuery(Generic[E, Unpack[A]]):
    bufs: dict[int, Tuple[list[jax.Array], list[jax.Array]]]
    component_data: list[Component]
    component_classes: list[type[Any]]
    inner: GraphQueryInner

    def __init__(
        self,
        inner: GraphQueryInner,
        component_data: list[Component],
        component_classes: list[type[Any]],
    ):
        self.bufs = inner.arrays()
        self.inner = inner
        self.component_data = component_data
        self.component_classes = component_classes

    @staticmethod
    def from_builder(
        new_tp: type[Any], builder: PipelineBuilder
    ) -> "GraphQuery[E, Any]":
        t_args = typing.get_args(new_tp)
        ids = []
        component_data = []
        component_classes = []
        edge_ty = t_args[0]
        edge_id = Component.id(edge_ty)
        for t_arg in t_args[1:]:
            component_classes.append(t_arg)
            data = t_arg.__metadata__[0]
            component_data.append(data)
            ids.append(Component.id(t_arg))
        return GraphQuery(
            GraphQueryInner.from_builder(builder, edge_id, ids),
            component_data,
            component_classes,
        )

    @staticmethod
    def init_builder(new_tp: type[Any], builder: PipelineBuilder):
        t_args = typing.get_args(new_tp)
        for t_arg in t_args:
            component_data: Component = t_arg.__metadata__[0]
            buf = builder.init_var(Component.id(t_arg), component_data.ty)

    def edge_fold(
        self, out_tp: type[O], init_value: O, fn: Callable[..., O]
    ) -> "Query[O]":
        out_bufs: list[jax.typing.ArrayLike] = []
        init_value_flat, init_value_tree = tree_flatten(init_value)
        for i, (f, to) in self.bufs.items():

            def vmap_inner(a):
                (f, to) = a

                def scan_inner(xs, to):
                    xs = tree_unflatten(init_value_tree, xs)
                    args = [
                        from_array(data, x)
                        for (x, data) in zip(f, self.component_classes)
                    ] + [
                        from_array(data, x)
                        for (x, data) in zip(to, self.component_classes)
                    ]
                    o = fn(xs, *args)
                    o_flat, _ = tree_flatten(o)
                    return (o_flat, 0)

                scan_out = jax.lax.scan(scan_inner, init_value_flat, to)[0]
                return scan_out

            buf = jax.vmap(vmap_inner)((f, to))
            (new_bufs, _) = tree_flatten(buf)
            if len(out_bufs) == 0:
                out_bufs = new_bufs
            else:
                out_bufs = [
                    jax.numpy.concatenate([x, y]) for (x, y) in zip(out_bufs, new_bufs)
                ]
            component_data = out_tp.__metadata__[0]  # type: ignore
        return Query(
            self.inner.map(out_bufs[0], component_data.to_metadata()),
            [component_data],
            [out_tp],
        )


class SystemParam(Protocol):
    @staticmethod
    def from_builder(builder: PipelineBuilder) -> Any: ...


class FromArray(Protocol):
    @staticmethod
    def from_array(arr: jax.Array) -> Any: ...


snake_case_pattern = re.compile(r"(?<!^)(?=[A-Z])")


class Archetype(Protocol):
    @classmethod
    def archetype_name(cls) -> str:
        return snake_case_pattern.sub("_", cls.__name__).lower()

    def component_data(self) -> list[Component]:
        return [
            v.__metadata__[0]
            for v in typing.get_type_hints(self, include_extras=True).values()
        ]

    def arrays(self) -> list[numpy.ndarray]:
        return [
            numpy.asarray(tree_flatten(v)[0][0])
            for (a, v) in self.__dict__.items()
            if not a.startswith("__") and not callable(getattr(self, a))
        ]


jax.tree_util.register_pytree_node(
    SpatialTransform,
    SpatialTransform.flatten,
    SpatialTransform.unflatten,
)
jax.tree_util.register_pytree_node(
    SpatialMotion, SpatialMotion.flatten, SpatialMotion.unflatten
)
jax.tree_util.register_pytree_node(
    SpatialForce, SpatialForce.flatten, SpatialForce.unflatten
)
jax.tree_util.register_pytree_node(
    SpatialInertia,
    SpatialInertia.flatten,
    SpatialInertia.unflatten,
)
jax.tree_util.register_pytree_node(Quaternion, Quaternion.flatten, Quaternion.unflatten)
jax.tree_util.register_pytree_node(Handle, Handle.flatten, Handle.unflatten)
jax.tree_util.register_pytree_node(Edge, Edge.flatten, Edge.unflatten)

WorldPos = Annotated[
    SpatialTransform, Component("world_pos", ComponentType.SpatialPosF64)
]
WorldVel = Annotated[
    SpatialMotion, Component("world_vel", ComponentType.SpatialMotionF64)
]
WorldAccel = Annotated[
    SpatialMotion, Component("world_accel", ComponentType.SpatialMotionF64)
]
Force = Annotated[SpatialForce, Component("force", ComponentType.SpatialMotionF64)]
Inertia = Annotated[SpatialInertia, Component("inertia", ComponentType.SpatialPosF64)]
PbrAsset = Annotated[Handle, Component(241, ComponentType.U64, "pbr_asset", True)]
EntityMetadataAsset = Annotated[
    Handle, Component(242, ComponentType.U64, "metadata_asset", True)
]
Seed = Annotated[jax.Array, Component("seed", ComponentType.U64)]
GizmoAsset = Annotated[Handle, Component(2243, ComponentType.U64, "gizmo_asset", True)]


class C:
    def __init__(self, tys: Union[tuple[Type], Type], values: Union[tuple[Any], Any]):
        if isinstance(tys, tuple) and isinstance(values, tuple):
            self.data = [ty.__metadata__[0] for ty in tys]  # type: ignore
            self.bufs = [numpy.asarray(tree_flatten(v)[0][0]) for v in values]
        else:
            self.data = [tys.__metadata__[0]]  # type: ignore
            self.bufs = [numpy.asarray(tree_flatten(values)[0][0])]

    @classmethod
    def archetype_name(cls) -> str:
        return snake_case_pattern.sub("_", cls.__name__).lower()

    def arrays(self):
        return self.bufs

    def component_data(self):
        return self.data


@dataclass
class Body(Archetype):
    world_pos: WorldPos = WorldPos.zero()
    world_vel: WorldVel = WorldVel.zero()
    inertia: Inertia = Inertia.from_mass(1.0)
    pbr: PbrAsset = Pbr(Mesh.sphere(1.0), Material.color(1.0, 1.0, 1.0))  # type: ignore # TODO(sphw): this code is wrong, but fixing it is hard
    force: Force = Force.zero()
    world_accel: WorldAccel = WorldAccel.zero()


def build_expr(builder: PipelineBuilder, sys: System) -> Any:
    sys.init(builder)

    def call(args, builder):
        builder.inject_args(args)
        sys.call(builder)

    xla = jax.xla_computation(lambda a: call(a, builder))(builder.var_arrays())
    return xla


class World(WorldBuilder):
    def run(
        self,
        sys: System,
        time_step: Optional[float] = None,
        client: Optional[Client] = None,
    ):
        current_frame = inspect.currentframe()
        if current_frame is None:
            raise Exception("No current frame")
        frame = current_frame.f_back
        if frame is None:
            raise Exception("No previous frame")
        addr = super().run(sys, time_step, client)
        locals = frame.f_locals
        if addr is not None:
            conduit_client = Conduit.tcp(addr)
            locals["client"] = conduit_client
            readline.set_completer(rlcompleter.Completer(locals).complete)
            readline.parse_and_bind("tab: complete")
            code.InteractiveConsole(locals=locals).interact()
