from dataclasses import dataclass
from typing import Protocol, TypeVar, Generic, Any, Self, Callable, Annotated
import typing
from enum import Enum
import jax
import inspect

class ComponentId:
    name = ""
    def __init__(self, name: str):
        self.name = name
class ComponentType(Enum):
    F32 = 0

class Component(Protocol):
    @staticmethod
    def component_id() -> ComponentId: ...
    @staticmethod
    def component_type() -> ComponentType: ...

def NewComponent(name: str, inner_tp: type[Component]):
    class Inner:
        inner: inner_tp
        @staticmethod
        def component_id():
            return ComponentId(name)
        @staticmethod
        def component_type():
            return inner_tp.component_type()
        def __init__(self, inner: inner_tp):
            self.inner = inner
    return Inner

class Scalar:
    @staticmethod
    def component_id() -> ComponentId:
        return ComponentId("Scalar")
    @staticmethod
    def component_type() -> ComponentType:
        return ComponentType.F32

Foo = NewComponent("Foo", Scalar)

foo = Foo(Scalar())
print(foo.component_id())
print(foo.component_type())

T = TypeVar('T', bound='Component')
Q = TypeVar('Q', bound='ComponentArray[Any]')
class ComponentArray(Generic[T]):
    buf: jax.Array
    @staticmethod
    def from_builder(new_tp: type[Q], builder) -> Q:
        t_arg = typing.get_args(new_tp)[0]
        print(t_arg.component_id().name)
        return new_tp()

#ComponentArray[Scalar].from_builder(Scalar, None)
ComponentArray.from_builder(ComponentArray[Scalar], None)

def system(func) -> Callable[[Any], None]:
    def inner(builder):
        sig = inspect.signature(func)
        params = sig.parameters
        args = [p.annotation.from_builder(p.annotation, builder) for (_, p) in params.items()]
        func(*args)
    return inner

@system
def test_system(a: ComponentArray[Scalar]):
    pass

test_system(None)


@dataclass
class ComponentData:
    id: ComponentId
    type: ComponentType

Bar = Annotated[Scalar, ComponentData(ComponentId("foo"), ComponentType.F32)]
#type Comp[T, Id, Ty] = Annotated[T, CompData(Id, Ty)]

def test(foo: Bar):
    pass

sig = inspect.signature(test)
params = sig.parameters
p = params['foo']
(m,) = p.annotation.__metadata__
print(m.type)

class Component:
    def __class_getitem__(cls, params):
        def parse_id(id):
            if id is str:
                return ComponentId(id)
            else:
                id
        if len(params) == 3:
            (t, id, type) = params
            return Annotated.__class_getitem__((t, ComponentData(id, type)))
        elif len(params) == 2:
            (t, id) = params
            type = t.__metadata__[0].type
            return Annotated.__class_getitem__((t, ComponentData(id, type)))
        else:
            raise Exception("Component must be called an ID and type")

Test = Component[Scalar, "foo", ComponentType.F32]
Test2 = Component[Test, "bar"]
