# ruff: noqa: F403
# ruff: noqa: F405

import code
import inspect
import re
import readline
import rlcompleter
import types
import typing
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import jax
import numpy
import pytest
from jax.tree_util import tree_flatten, tree_unflatten
from typing_extensions import TypeVarTuple, Unpack

from .elodin import *

__doc__ = elodin.__doc__

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


def pytest_generate_tests(metafunc):
    if "df" in metafunc.fixturenames and _has_df_key not in metafunc.definition.stash:
        metafunc.definition.stash[_has_df_key] = True
        path = metafunc.config.getoption("batch_results")
        dfs, sample_numbers = read_batch_results(path)
        metafunc.parametrize("df", dfs, ids=sample_numbers)


def system(func) -> System:
    sig = inspect.signature(func)
    # TODO(sphw): use varadict function
    params = sig.parameters
    input_ids = []
    edge_ids = []
    for _, p in params.items():
        input_ids.extend(p.annotation.component_ids(p.annotation))
        if getattr(p.annotation, "edge_ids", None) is not None:
            edge_ids.extend(p.annotation.edge_ids(p.annotation))
    output_ids = []
    return_annotation = sig.return_annotation
    if return_annotation is not inspect._empty:
        output_ids.extend(sig.return_annotation.component_ids(return_annotation))

    def outer(builder):
        def inner(*args):
            new_args = []
            for _, p in params.items():
                new_args.append(p.annotation.from_builder(p.annotation, builder, args))
            output = func(*new_args)
            return output.output(builder, args)

        return inner

    return PyFnSystem(outer, input_ids, output_ids, edge_ids, func.__repr__()).system()


T = TypeVar("T")
S = TypeVarTuple("S")
A = TypeVarTuple("A")
B = TypeVarTuple("B")


class Query(Generic[Unpack[A]]):
    bufs: list[jax.Array]
    component_data: list[Metadata]
    component_classes: list[type[Any]]
    inner: QueryInner

    def __init__(
        self,
        inner: QueryInner,
        component_data: list[Metadata],
        component_classes: list[type[Any]],
    ):
        self.bufs = inner.arrays()
        self.inner = inner
        self.component_data = component_data
        self.component_classes = component_classes

    def map(
        self,
        out_tps: Union[Tuple[Annotated[Any, Component], ...], Annotated[Any, Component]],
        f: Callable[[Unpack[A]], Union[Tuple[Unpack[S]], T]],
    ) -> "Query[Unpack[S]]":
        out_tps_tuple: Tuple[Annotated[Any, Component], ...] = (
            (out_tps,) if not isinstance(out_tps, tuple) else out_tps
        )
        buf = jax.vmap(
            lambda b: f(
                *[from_array(cls, x) for (x, cls) in zip(b, self.component_classes)]  # type: ignore
            ),
            in_axes=0,
            out_axes=0,
        )(self.bufs)
        (bufs, _) = tree_flatten(buf)
        inner = None
        component_data = []
        component_classes = []
        for out_tp, buf in zip(out_tps_tuple, bufs):
            this_inner = self.inner.map(buf, Metadata.of(out_tp))  # type: ignore
            if inner is None:
                inner = this_inner
            else:
                inner = inner.join_query(this_inner)
            component_data += [Metadata.of(out_tp)]  # type: ignore
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
    def component_ids(new_tp: type[Any]) -> list[str]:
        t_args = typing.get_args(new_tp)
        ids = []
        for t_arg in t_args:
            ids.append(Component.name(t_arg))
        return ids

    @staticmethod
    def from_builder(new_tp: type[Any], builder: SystemBuilder, args: list[Any]) -> "Query[Any]":
        t_args = typing.get_args(new_tp)
        ids = []
        component_data = []
        component_classes = []
        for t_arg in t_args:
            component_data.append(Metadata.of(t_arg))
            component_classes.append(t_arg)
            ids.append(Component.name(t_arg))
        return Query(
            QueryInner.from_builder(builder, ids, args),
            component_data,
            component_classes,
        )

    def output(self, builder: SystemBuilder, args: list[Any]) -> Any:
        return self.inner.output(builder, args)

    def __getitem__(self, index: int) -> Any:
        if len(self.bufs) > 1:
            raise Exception("Cannot index into a query with multiple inputs")
        cls = self.component_classes[0]
        return from_array(cls, self.bufs[0][index])


def map(
    func: Callable[..., Union[Tuple[Annotated[Any, Component], ...], Annotated[Any, Component]]],
) -> System:
    sig = inspect.signature(func)
    tys = list(sig.parameters.values())
    query = Query[tuple(ty.annotation for ty in tys)]  # type: ignore
    return_ty = sig.return_annotation
    if isinstance(return_ty, types.GenericAlias):
        return_ty = tuple(return_ty.__args__)

    @system
    def inner(q: query) -> Query[return_ty]:  # type: ignore
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


class RevEdge: ...


class TotalEdge: ...


class GraphQuery(Generic[E]):
    bufs: dict[int, Tuple[list[jax.Array], list[jax.Array]]]
    inner: GraphQueryInner

    def __init__(
        self,
        inner: GraphQueryInner,
    ):
        self.inner = inner

    @staticmethod
    def from_builder(new_tp: type[Any], builder: SystemBuilder, _: list[Any]) -> "GraphQuery[E]":
        t_args = typing.get_args(new_tp)
        edge_ty = t_args[0]
        if isinstance(edge_ty, type(TotalEdge)):
            return GraphQuery(GraphQueryInner.from_builder_total_edge(builder))
        edge_id = Component.name(edge_ty)
        reverse = False
        if len(edge_ty.__metadata__) > 1 and edge_ty.__metadata__[1] is RevEdge:
            reverse = True
        return GraphQuery(
            GraphQueryInner.from_builder(builder, edge_id, reverse),
        )

    @staticmethod
    def component_ids(_: type[Any]) -> list[str]:
        return []

    @staticmethod
    def edge_ids(new_tp: type[Any]) -> list[str]:
        t_args = typing.get_args(new_tp)
        ids = []
        for t_arg in t_args:
            if t_arg is not TotalEdge:
                ids.append(Component.name(t_arg))
        return ids

    def edge_fold(
        self,
        from_query: Query[Unpack[A]],
        to_query: Query[Unpack[B]],
        out_tp: Annotated[Any, Component],
        init_value: T,
        fn: Callable[..., T],
    ) -> "Query[T]":
        out_bufs: list[jax.typing.ArrayLike] = []
        bufs = self.inner.arrays(from_query.inner, to_query.inner)
        init_value_flat, init_value_tree = tree_flatten(init_value)
        for _, (f, to) in bufs.items():

            def vmap_inner(a):
                (f, to) = a

                def scan_inner(xs, to):
                    xs = tree_unflatten(init_value_tree, xs)
                    args = [
                        from_array(data, x) for (x, data) in zip(f, from_query.component_classes)
                    ] + [from_array(data, x) for (x, data) in zip(to, to_query.component_classes)]
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
        component_data = Metadata.of(out_tp)
        return Query(
            self.inner.map(
                from_query.inner,
                to_query.inner,
                out_bufs[0],
                component_data,
            ),
            [component_data],
            [out_tp],
        )


snake_case_pattern = re.compile(r"(?<!^)(?=[A-Z])")


class Archetype(Protocol):
    @classmethod
    def archetype_name(cls) -> str:
        return snake_case_pattern.sub("_", cls.__name__).lower()

    def component_data(self) -> list[Metadata]:
        return [Metadata.of(v) for v in typing.get_type_hints(self, include_extras=True).values()]

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
jax.tree_util.register_pytree_node(SpatialMotion, SpatialMotion.flatten, SpatialMotion.unflatten)
jax.tree_util.register_pytree_node(SpatialForce, SpatialForce.flatten, SpatialForce.unflatten)
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
        metadata={"element_names": "q0,q1,q2,q3,x,y,z", "priority": 5},
    ),
]
WorldVel = Annotated[
    SpatialMotion,
    Component(
        "world_vel",
        metadata={"element_names": "ωx,ωy,ωz,x,y,z", "priority": 5},
    ),
]
WorldAccel = Annotated[
    SpatialMotion,
    Component(
        "world_accel",
        metadata={"element_names": "αx,αy,αz,x,y,z", "priority": 5},
    ),
]
Force = Annotated[
    SpatialForce,
    Component(
        "force",
        metadata={"element_names": "τx,τy,τz,x,y,z", "priority": 5},
    ),
]
Inertia = Annotated[
    SpatialInertia,
    Component("inertia", metadata={"priority": 5}),
]
Seed = Annotated[jax.Array, Component("seed", ComponentType.U64, metadata={"priority": 5})]
SimulationTick = Annotated[
    jax.Array, Component("simulation_tick", ComponentType.F64, metadata={"priority": 7})
]
SimulationTimeStep = Annotated[
    jax.Array,
    Component("simulation_time_step", ComponentType.F64, metadata={"priority": 8}),
]
MeshAsset = Annotated[
    Handle,
    Component("asset_handle_mesh", asset=True, metadata={"priority": -1}),
]
MaterialAsset = Annotated[
    Handle,
    Component("asset_handle_material", asset=True, metadata={"priority": -1}),
]
GlbAsset = Annotated[
    Handle,
    Component("asset_handle_glb", asset=True, metadata={"priority": -1}),
]
GizmoAsset = Annotated[
    Handle,
    Component("asset_handle_gizmo", asset=True, metadata={"priority": -1}),
]
PanelAsset = Annotated[
    Handle,
    Component("asset_handle_panel", asset=True, metadata={"priority": -1}),
]
Camera = Annotated[
    jax.Array,
    Component(
        "camera",
        ComponentType(PrimitiveType.U64, (1,)),
    ),
]


class C:
    def __init__(self, tys: Union[tuple[Type], Type], values: Union[tuple[Any], Any]):
        if isinstance(tys, tuple) and isinstance(values, tuple):
            self.data = [Metadata.of(ty) for ty in tys]  # type: ignore
            self.bufs = [numpy.asarray(tree_flatten(v)[0][0]) for v in values]
        else:
            self.data = [Metadata.of(tys)]  # type: ignore
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
    inertia: Inertia = SpatialInertia(1.0)
    force: Force = SpatialForce.zero()
    world_accel: WorldAccel = SpatialMotion.zero()


@dataclass
class Shape(Archetype):
    mesh: MeshAsset
    material: MaterialAsset


@dataclass
class Scene(Archetype):
    glb: GlbAsset


class World(WorldBuilder):
    def run(
        self,
        system: System,
        time_step: Optional[float] = None,
        sim_time_step: Optional[float] = None,
        run_time_step: Optional[float] = None,
        output_time_step: Optional[float] = None,
        max_ticks: Optional[int] = None,
        client: Optional[Client] = None,
    ):
        current_frame = inspect.currentframe()
        if current_frame is None:
            raise Exception("No current frame")
        frame = current_frame.f_back
        if frame is None:
            raise Exception("No previous frame")
        if sim_time_step is None:
            sim_time_step = time_step
        addr = super().run(
            system, sim_time_step, run_time_step, output_time_step, max_ticks, client
        )
        locals = frame.f_locals
        if addr is not None:
            impeller_client = Impeller.tcp(addr)
            locals["client"] = impeller_client
            readline.set_completer(rlcompleter.Completer(locals).complete)
            readline.parse_and_bind("tab: complete")
            code.InteractiveConsole(locals=locals).interact()

    def serve(
        self,
        system: System,
        addr: Optional[str] = None,
        time_step: Optional[float] = None,
        sim_time_step: Optional[float] = None,
        run_time_step: Optional[float] = None,
        output_time_step: Optional[float] = None,
        max_ticks: Optional[int] = None,
        client: Optional[Client] = None,
    ):
        if sim_time_step is None:
            sim_time_step = time_step
        super().serve(
            system,
            False,
            sim_time_step,
            run_time_step,
            output_time_step,
            max_ticks,
            client,
            addr,
        )

    def view(
        self,
        system: System,
        time_step: Optional[float] = None,
        sim_time_step: Optional[float] = None,
        run_time_step: Optional[float] = None,
        output_time_step: Optional[float] = None,
        client: Optional[Client] = None,
    ) -> Any:
        from IPython.display import IFrame

        if sim_time_step is None:
            sim_time_step = time_step
        addr = super().serve(
            system,
            True,
            sim_time_step,
            run_time_step,
            output_time_step,
            None,
            client,
            None,
        )
        return IFrame(f"http://{addr}", width=960, height=540)

    def glb(self, url: str) -> Scene:
        return Scene(self.insert_asset(Glb(url)))  # type: ignore

    def shape(self, mesh: Mesh, material: Material) -> Shape:
        return Shape(self.insert_asset(mesh), self.insert_asset(material))  # type: ignore
