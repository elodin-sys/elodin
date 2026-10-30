"""Pure functional stepping for an elodin world (stand-in until JaxSim grows a step_fn).

`JaxStep(world, system)` compiles the world with `world.to_jax_func` and exposes
`step(state) -> state` over a tuple of arrays (one per component slot), so rollouts can use
jax.jit, jax.vmap, lax.scan and jax.grad. Components that are read but never written are not
outputs of the compiled function; they pass through unchanged (JaxSim.generate_index_map
raises on such worlds, which is why this calls to_jax_func directly).
"""

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np


class JaxStep:
    def __init__(self, world: el.World, system: el.System, simulation_rate: float = 300.0):
        py_sim, ins, outs, state, names, entities, comp_entities = world.to_jax_func(
            system, simulation_rate, 1.0, None
        )
        self.py_sim = py_sim
        self.names: list[str] = [names[i] for i in ins]
        self.entity_ids: dict[str, int] = dict(entities)
        self.component_entities: dict[str, list[int]] = dict(comp_entities)
        out_pos = {cid: c for c, cid in enumerate(outs)}
        # src[d] = output index feeding input slot d, or None when the slot is read-only
        self.src = [out_pos.get(cid) for cid in ins]
        self.init = tuple(jnp.asarray(np.asarray(a)) for a in state)
        self.dtypes = [a.dtype for a in self.init]
        self._slots = {n: d for d, n in enumerate(self.names)}
        self.tick_slot = self._slots.get("tick")

    def slot(self, name: str) -> int:
        return self._slots[name]

    def has(self, name: str) -> bool:
        return name in self._slots

    def step(self, state: tuple) -> tuple:
        state = list(state)
        if self.tick_slot is not None:
            t = self.tick_slot
            state[t] = state[t] + jnp.ones((), state[t].dtype)
        out = self.py_sim(*state)
        return tuple(
            state[d] if c is None else jnp.asarray(out[c]).astype(self.dtypes[d])
            for d, c in enumerate(self.src)
        )

    def run(self, state: tuple, ticks: int, unroll: int = 1) -> tuple:
        def body(s, _):
            return self.step(s), None

        return jax.lax.scan(body, state, None, length=ticks, unroll=unroll)[0]

    def get(self, state: tuple, name: str) -> jax.Array:
        return state[self._slots[name]]

    def set(self, state: tuple, name: str, value) -> tuple:
        d = self._slots[name]
        state = list(state)
        state[d] = jnp.broadcast_to(jnp.asarray(value, self.dtypes[d]), state[d].shape)
        return tuple(state)

    def entity_index(self, name: str, entity: str) -> int | None:
        """Row of `entity` in a multi-entity slot, or None when the slot has no entity axis."""
        ids = self.component_entities[name]
        if len(ids) == 1:
            return None
        return ids.index(self.entity_ids[entity])
