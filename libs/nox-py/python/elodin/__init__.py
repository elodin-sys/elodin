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
if hasattr(elodin, "__all__"):
    __all__ = elodin.__all__

Self = TypeVar("Self")
class System(Protocol):
    @staticmethod
    def call(builder: PipelineBuilder): ...
    def init(builder: PipelineBuilder) -> PipelineBuilder: ...
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
            for (_, p) in params.items():
                p.annotation.init_builder(p.annotation, builder)
            if sig.return_annotation is not inspect._empty:
                sig.return_annotation.init_builder(sig.return_annotation, builder)

        def call(self, builder):
            sig = inspect.signature(func)
            params = sig.parameters
            args = [p.annotation.from_builder(p.annotation, builder) for (_, p) in params.items()]
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

O = TypeVar('O')
T = TypeVar('T', bound='Union[jax.Array, FromArray]')
Q = TypeVar('Q', bound='ComponentArray[Any]')
class ComponentArray(Generic[T]):
    buf: jax.Array
    component_data: ComponentData
    metadata: ComponentArrayMetadata


    @staticmethod
    def from_builder(new_tp: type[Q], builder: PipelineBuilder) -> Q:
        t_arg = typing.get_args(new_tp)[0]
        arr = new_tp()
        arr.component_data = t_arg.__metadata__[0]
        (metadata, buf) = builder.get_var(arr.component_data.id)
        arr.metadata = metadata
        arr.buf = buf
        return arr

    @staticmethod
    def init_builder(new_tp: Type[Q], builder: PipelineBuilder):
        t_arg = typing.get_args(new_tp)[0]
        component_data: ComponentData = t_arg.__metadata__[0]
        buf = builder.init_var(component_data.id, component_data.type )

    def insert_into_builder(self, builder: PipelineBuilder):
        builder.set_var(self.component_data.id, self.metadata, self.buf)

    def map(self, f: Callable[[T], O]) -> Q:
        buf = jax.vmap(lambda b: f(self.component_data.from_expr(b)))(self.buf)
        arr = ComponentArray[O]()
        arr.metadata = self.metadata
        arr.buf = buf
        arr.component_data = self.component_data
        return arr
    def join(self, other: Q) -> Any:
        (metadata, bufs) = self.metadata.join(self.buf, other.metadata, [other.buf])
        q = Query()
        q.bufs = bufs
        q.component_data = [self.component_data, other.component_data]
        q.metadata = metadata
        return q

A = TypeVarTuple('A')
B = TypeVar('B', bound='Query[Any]')
class Query(Generic[Unpack[A]]):
    bufs: list[jax.Array]
    component_data: list[ComponentData]
    metadata: ComponentArrayMetadata
    def map(self, out_tp: type[O], f: Callable[[*A], O]) -> Q:
        buf = jax.vmap(lambda b: f(*[data.from_expr(x) for (x, data) in zip(b, self.component_data)]) )(self.bufs)
        arr = ComponentArray[O]()
        arr.buf = buf
        arr.component_data = out_tp.__metadata__[0]
        arr.metadata = self.metadata
        return arr

    def join(self, other: ComponentArray[O]) -> B:
        (metadata, bufs) = other.metadata.join(other.buf, self.metadata, self.bufs)
        q = Query()
        q.bufs = bufs
        q.component_data = self.component_data
        q.component_data.append(other.component_data)
        q.metadata = metadata
        return q

    @staticmethod
    def from_builder(new_tp: type[B], builder: PipelineBuilder) -> Q:
        t_args = typing.get_args(new_tp)
        query = None
        for t_arg in t_args:
            arr = ComponentArray.from_builder(ComponentArray[t_arg], builder)
            if query is None:
                query = Query()
                query.component_data = [arr.component_data]
                query.bufs = [arr.buf]
                query.metadata = arr.metadata
            else:
                query = query.join(arr)
        return query

    @staticmethod
    def init_builder(new_tp: Type[B], builder: PipelineBuilder):
        t_args = typing.get_args(new_tp)
        for t_arg in t_args:
            component_data: ComponentData = t_arg.__metadata__[0]
            buf = builder.init_var(component_data.id, component_data.type )




class SystemParam(Protocol):
    @staticmethod
    def from_builder(builder: PipelineBuilder) -> Any: ...

class FromArray(Protocol):
    @staticmethod
    def from_array(arr: jax.Array) -> Any: ...


class Component:
    def __class_getitem__(cls, params):
        def parse_id(id):
            if isinstance(id, str):
                return ComponentId(id)
            else:
                return id
        def from_expr(ty):
            if ty is jax.Array:
                return lambda x: x
            else:
                return ty.from_array
        if len(params) == 4:
            (t, id, type, asset) = params
            id = parse_id(id)
            return Annotated.__class_getitem__((t, ComponentData(id, type, asset, from_expr(t)))) # type: ignore
        if len(params) == 3:
            (t, id, type) = params
            id = parse_id(id)
            return Annotated.__class_getitem__((t, ComponentData(id, type, False, from_expr(t)))) # type: ignore
        elif len(params) == 2:
            (t, id) = params
            id = parse_id(id)
            type = t.__metadata__[0].type
            return Annotated.__class_getitem__((t, ComponentData(id, type, from_expr(t)))) # type: ignore
        else:
            raise Exception("Component must be called an ID and type")

class Archetype(Protocol):
    def archetype_id(self) -> int:
        return abs(hash(type(self).__name__))
    def component_data(self) -> list[ComponentData]:
        return [v.__metadata__[0] for v in typing.get_type_hints(self, include_extras=True).values()]
    def arrays(self) -> list[jax.Array]:
        return [numpy.asarray(tree_flatten(v)[0][0]) for (a, v) in self.__dict__.items() if not a.startswith('__') and not callable(getattr(self, a))]

jax.tree_util.register_pytree_node(elodin.SpatialTransform,elodin.SpatialTransform.flatten,elodin.SpatialTransform.unflatten)
jax.tree_util.register_pytree_node(elodin.SpatialMotion,elodin.SpatialMotion.flatten,elodin.SpatialMotion.unflatten)
jax.tree_util.register_pytree_node(elodin.SpatialForce,elodin.SpatialForce.flatten,elodin.SpatialForce.unflatten)
jax.tree_util.register_pytree_node(elodin.SpatialInertia,elodin.SpatialInertia.flatten,elodin.SpatialInertia.unflatten)
jax.tree_util.register_pytree_node(elodin.Quaternion,elodin.Quaternion.flatten,elodin.Quaternion.unflatten)
jax.tree_util.register_pytree_node(elodin.Handle,elodin.Handle.flatten,elodin.Handle.unflatten)

WorldPos = Component[SpatialTransform, "world_pos", ComponentType.SpatialPosF64]
WorldVel = Component[SpatialMotion, "world_vel", ComponentType.SpatialMotionF64]
WorldAccel = Component[SpatialMotion, "world_accel", ComponentType.SpatialMotionF64]
Force = Component[SpatialForce, "force", ComponentType.SpatialMotionF64]
Inertia = Component[SpatialInertia, "inertia", ComponentType.SpatialPosF64]
MeshAsset = Component[Handle, "mesh", ComponentType.U64, True]
MaterialAsset = Component[Handle, "material", ComponentType.U64, True]

@dataclass
class Body(Archetype):
    world_pos: WorldPos
    world_vel: WorldVel
    world_accel: WorldAccel
    force: Force
    inertia: Inertia
    mesh: MeshAsset
    material: MaterialAsset


def build_expr(builder: PipelineBuilder, sys: System) -> Any:
    sys.init(builder)
    def call(args, builder):
        builder.inject_args(args)
        sys.call(builder)
    xla = jax.xla_computation(lambda a: call(a, builder))(builder.var_arrays())
    return xla

def dbg(x):
    print(x)
    return x
