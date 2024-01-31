import nox_py
import jax
import jax.numpy as np
from nox_py import Component, ComponentType, system, ComponentArray, Archetype, WorldBuilder, Client, ComponentId
from dataclasses import dataclass

X = Component[jax.Array, "x", ComponentType.F32]
Y = Component[jax.Array, "y", ComponentType.F32]
print(X.__metadata__)

@system
def foo(x: ComponentArray[X]) -> ComponentArray[X]:
  return x

@system
def bar(y: ComponentArray[Y], x: ComponentArray[X]) -> ComponentArray[X]:
  x.buf += y.buf
  return x

sys = foo.pipe(bar)


@dataclass
class Test(Archetype):
  x: X
  y: Y


client = Client.cpu()
w = WorldBuilder()
w.spawn(Test(np.array([1.0], dtype='float32'), np.array([500.0], dtype='float32')))
w.spawn(Test(np.array([5.0], dtype='float32'), np.array([500.0], dtype='float32')))
exec = w.build(sys, client)
exec.run(client)
print(exec.column(ComponentId("y")))
print(exec.column(ComponentId("x")))
