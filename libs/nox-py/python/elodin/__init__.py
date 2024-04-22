from .elodin import *
import types
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
import polars as pl
import pytest


__doc__ = elodin.__doc__  # type: ignore

_called_from_test = False
_has_df_key = pytest.StashKey[str]()

jax.config.update("jax_enable_x64", True)

Self = TypeVar("Self")


def pytest_addoption(parser):
    parser.addoption(
        "--batch-results",
        action="store",
        default="",
        help="path to batch results directory",
    )


def pytest_configure(config):
    global _called_from_test
    _called_from_test = True


def pytest_unconfigure(config):
    global _called_from_test
    _called_from_test = False


def read_sample_results(path):
    df = read_batch_results(path)
    sample_numbers = df["sample_number"].unique().sort()
    dfs = [df.filter(pl.col("sample_number") == s) for s in sample_numbers]
    return dfs


def sample_number(df):
    sample_number = df["sample_number"][0]
    return "sample_number=" + str(sample_number)


def pytest_generate_tests(metafunc):
    if "df" in metafunc.fixturenames and _has_df_key not in metafunc.definition.stash:
        metafunc.definition.stash[_has_df_key] = True
        path = metafunc.config.getoption("batch_results")
        dfs = read_sample_results(path)
        metafunc.parametrize("df", dfs, ids=sample_number)


class System(Protocol):
    def call(self, builder: PipelineBuilder):
        ...

    def init(self, builder: PipelineBuilder):
        ...

    def pipe(self, other: Any) -> Any:
        return Pipe(self, other)

    def __or__(self, other: Any) -> Any:
        return self.pipe(other)

@dataclass
class Pipe(System):
    a: System
    b: System

    def init(self, builder: PipelineBuilder):
        self.a.init(builder)
        self.b.init(builder)

    def call(self, builder: PipelineBuilder):
        self.a.call(builder)
        self.b.call(builder)


def system(func) -> System:
    class Inner(System):
        func: Callable[[Any], Any]

        def init(self, builder: PipelineBuilder):
            sig = inspect.signature(func)
            params = sig.parameters
            for _, p in params.items():
                p.annotation.init_builder(p.annotation, builder)
            if sig.return_annotation is not inspect._empty:
                sig.return_annotation.init_builder(sig.return_annotation, builder)

        def call(self, builder: PipelineBuilder):
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

S = TypeVarTuple("S")

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

    def map(self, out_tps: Union[Tuple[Annotated[Any, Component], ...], Annotated[Any, Component]], f: Callable[[Unpack[A]], Union[Tuple[Unpack[S]], O]]) -> 'Query[Unpack[S]]':
        out_tps_tuple: Tuple[Annotated[Any, Component], ...] = (out_tps,) if not isinstance(out_tps, tuple)  else out_tps
        buf = jax.vmap(
            lambda b: f(
                *[from_array(cls, x) for (x, cls) in zip(b, self.component_classes)] # type: ignore
            ),
            in_axes=0,
            out_axes=0,
        )(self.bufs)
        (bufs, _) = tree_flatten(buf)
        inner = None
        component_data = []
        component_classes = []
        for out_tp, buf in zip(out_tps_tuple , bufs):
            this_inner = self.inner.map(buf, out_tp.__metadata__[0].to_metadata()) # type: ignore
            if inner is None:
                inner = this_inner
            else:
                inner = inner.join_query(this_inner)
            component_data += [out_tp.__metadata__[0]] # type: ignore
            component_classes += [out_tp]
        if inner is None:
            raise Exception("query returned no components")
        return Query(inner, component_data, component_classes)

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
            builder.init_var(Component.id(t_arg), component_data.ty)

    def __getitem__(self, index: int) -> Any:
        if len(self.bufs) > 1:
            raise Exception("Cannot index into a query with multiple inputs")
        cls = self.component_classes[0]
        return from_array(cls, self.bufs[0][index])

    def insert_into_builder(self, builder: PipelineBuilder):
        self.inner.insert_into_builder(builder)


def map(func: Callable[..., Union[Tuple[Annotated[Any, Component], ...], Annotated[Any, Component]]]) -> System:
    sig = inspect.signature(func)
    tys = list(sig.parameters.values())
    query = Query[tuple(ty.annotation for ty in tys)] # type: ignore
    return_ty = sig.return_annotation
    if isinstance(return_ty, types.GenericAlias):
        return_ty = tuple(return_ty.__args__)
    @system
    def inner(q: query): # type: ignore
        return q.map(return_ty, func)
    return inner



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
            builder.init_var(Component.id(t_arg), component_data.ty)
    def edge_fold(self, out_tp: Annotated[Any, Component], init_value: O, fn: Callable[..., O]) -> 'Query[O]':
        out_bufs: list[jax.typing.ArrayLike] = []
        init_value_flat, init_value_tree = tree_flatten(init_value)
        for (_, (f, to)) in self.bufs.items():
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
                out_bufs = [jax.numpy.concatenate([x, y]) for (x, y) in zip(out_bufs, new_bufs)]
        component_data = out_tp.__metadata__[0] # type: ignore
        return Query(self.inner.map(out_bufs[0], component_data.to_metadata()), [component_data], [out_tp])


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
    SpatialTransform,
    Component(
        "world_pos",
        ComponentType.SpatialPosF64,
        metadata={"element_names": "q0,q1,q2,q3,x,y,z", "priority": 5},
    ),
]
WorldVel = Annotated[
    SpatialMotion,
    Component(
        "world_vel",
        ComponentType.SpatialMotionF64,
        metadata={"element_names": "ωx,ωy,ωz,x,y,z", "priority": 5},
    ),
]
WorldAccel = Annotated[
    SpatialMotion,
    Component(
        "world_accel",
        ComponentType.SpatialMotionF64,
        metadata={"element_names": "αx,αy,αz,x,y,z", "priority": 5},
    ),
]
Force = Annotated[
    SpatialForce,
    Component(
        "force",
        ComponentType.SpatialMotionF64,
        metadata={"element_names": "τx,τy,τz,x,y,z", "priority": 5},
    ),
]
Inertia = Annotated[
    SpatialInertia,
    Component("inertia", ComponentType.SpatialPosF64, metadata={"priority": 5}),
]
Seed = Annotated[
    jax.Array, Component("seed", ComponentType.U64, metadata={"priority": 5})
]
Time = Annotated[
    jax.Array, Component("time", ComponentType.F64, metadata={"priority": 5})
]
PbrAsset = Annotated[
    Handle, Component(241, ComponentType.U64, True, metadata={"name": "pbr_asset", "priority": -1})
]
EntityMetadataAsset = Annotated[
    Handle, Component(242, ComponentType.U64, True, metadata={"name": "metadata_asset", "priority": -1})
]
GizmoAsset = Annotated[
    Handle, Component(2243, ComponentType.U64, True, metadata={"name": "gizmo_asset", "priority": -1})
]
PanelAsset = Annotated[
    Handle, Component(2244, ComponentType.U64, True, metadata={"name": "panel_asset", "priority": -1})
]


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
    world_pos: WorldPos = SpatialTransform.zero()
    world_vel: WorldVel = SpatialMotion.zero()
    inertia: Inertia = SpatialInertia.from_mass(1.0)
    pbr: PbrAsset = Pbr(Mesh.sphere(1.0), Material.color(1.0, 1.0, 1.0)) # type: ignore # TODO(sphw): this code is wrong, but fixing it is hard
    force: Force = SpatialForce.zero()
    world_accel: WorldAccel = SpatialMotion.zero()


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
        system: System,
        time_step: Optional[float] = None,
        client: Optional[Client] = None,
    ):
        current_frame = inspect.currentframe()
        if current_frame is None:
            raise Exception("No current frame")
        frame = current_frame.f_back
        if frame is None:
            raise Exception("No previous frame")
        addr = super().run(system, time_step, client)
        locals = frame.f_locals
        if addr is not None:
            conduit_client = Conduit.tcp(addr)
            locals["client"] = conduit_client
            readline.set_completer(rlcompleter.Completer(locals).complete)
            readline.parse_and_bind("tab: complete")
            code.InteractiveConsole(locals=locals).interact()

    def serve(
        self,
        system: System,
        addr: Optional[str] = None,
        time_step: Optional[float] = None,
        client: Optional[Client] = None,
    ):
        super().serve(system, False, time_step, client, addr)

    def view(self, system: System, time_step: Optional[float] = None, client: Optional[Client] = None) -> Any:
        from IPython.display import IFrame, display
        addr = super().serve(system, True, time_step, client, None)
        return IFrame(f"http://{addr}", width=960, height=540)
