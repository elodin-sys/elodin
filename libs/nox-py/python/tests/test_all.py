import pytest
import elodin
import jax
import jax.numpy as np
from elodin import Component, ComponentType, system, ComponentArray, Archetype, WorldBuilder, Client, ComponentId, Query
from dataclasses import dataclass

def test_basic_system():
    X = Component[jax.Array, "x", ComponentType.F32]
    Y = Component[jax.Array, "y", ComponentType.F32]
    E = Component[jax.Array, "e", ComponentType.F32]

    @system
    def foo(x: ComponentArray[X]) -> ComponentArray[X]:
        return x.map(lambda x: x * 2)
    @system
    def bar(q: Query[X, Y]) -> ComponentArray[X]:
        return q.map(X, lambda x, y: x * y)
    @system
    def baz(q: Query[X, E]) -> ComponentArray[X]:
        return q.map(X, lambda x, e: x + e)

    @dataclass
    class Test(Archetype):
        x: X
        y: Y

    @dataclass
    class Effect(Archetype):
        e: E

    sys = foo.pipe(bar).pipe(baz)
    client = Client.cpu()
    w = WorldBuilder()
    w.spawn(Test(np.array([1.0], dtype='float32'), np.array([500.0], dtype='float32')))
    id = w.spawn(Test(np.array([15.0], dtype='float32'), np.array([500.0], dtype='float32')))
    w.spawn_with_entity_id(Effect(np.array([15.0], dtype='float32')), id)
    exec = w.build(sys)
    exec.run(client)
    x1 = exec.column_array(ComponentId("x"))
    y1 = exec.column_array(ComponentId("y"))
    assert (x1 == [1000.0, 15015.0]).all()
    assert (y1 == [500.0, 500.0]).all()
    exec.run(client)
    x1 = exec.column_array(ComponentId("x"))
    y1 = exec.column_array(ComponentId("y"))
    assert (x1 == [1000000., 15015015.]).all()
    assert (y1 == [500.0, 500.0]).all()
