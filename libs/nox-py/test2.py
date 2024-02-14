import elodin
import jax
import jax.numpy as np
from elodin import Component, ComponentType, system, ComponentArray, Archetype, WorldBuilder, Client, ComponentId, Query
from dataclasses import dataclass

X = Component[jax.Array, "x", ComponentType.F32]
Y = Component[jax.Array, "y", ComponentType.F32]
E = Component[jax.Array, "e", ComponentType.F32]
print(X.__metadata__)

@system
def foo(x: ComponentArray[X]) -> ComponentArray[X]:
  return x.map(lambda x: x * 2)

# @system
# def bar(y: ComponentArray[Y], x: ComponentArray[X]) -> ComponentArray[X]:
#   return y.join(x).map(X, lambda x, y: x * y)

@system
def bar(q: Query[X, Y]) -> ComponentArray[X]:
  return q.map(X, lambda x, y: x * y)

@system
def baz(q: Query[X, E]) -> ComponentArray[X]:
  return q.map(X, lambda x, e: x + e)


sys = foo.pipe(bar).pipe(baz)


@dataclass
class Test(Archetype):
  x: X
  y: Y

@dataclass
class Effect(Archetype):
  e: E


client = Client.cpu()
w = WorldBuilder()
w.spawn(Test(np.array([1.0], dtype='float32'), np.array([500.0], dtype='float32')))
id = w.spawn(Test(np.array([15.0], dtype='float32'), np.array([500.0], dtype='float32')))
w.spawn_with_entity_id(Effect(np.array([15.0], dtype='float32')), id)
exec = w.build(sys, client)
exec.run(client)
y1 = exec.column_array(ComponentId("y"))
x1 = exec.column_array(ComponentId("x"))
print(y1)
print(x1)
exec.run(client)
print(exec.column_array(ComponentId("y")))
print(exec.column_array(ComponentId("x")))
print(y1)
print(x1)
