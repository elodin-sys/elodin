from jax import numpy as np
from jax.tree_util import register_pytree_node


class JaxSim:
    """
    A class to simulate a system within a given world context.

    `step`, `get_state` and `set_state` act on `self.state`. `step_fn`, `get` and `set` are
    their pure counterparts: they take a state and return a new one, so they compose with
    `jax.jit`, `jax.vmap` and `jax.lax.scan`.

    Attributes:
    py_sim : JAX simulation instance
        The simulation instance created from the world and system.
    inputs : list
        The input parameters for the simulation.
    outputs : list
        The output parameters from the simulation.
    state : list
        The current state of the simulation.
    dictionary : dict
        A dictionary mapping component IDs to their names.
    entity_dict : dict
        A dictionary mapping entity names to their IDs.
    component_entity_dict : dict
        A dictionary mapping component names to a list of their entity IDs.
    map : list
        For each input, the output that updates it (None when no system writes it), its
        position in the state and its component name.
    """

    def __init__(
        self,
        sim_obj,
        inputs,
        outputs,
        state,
        dictionary,
        entity_dict,
        component_entity_dict,
        map=None,
        workers=1,
        simulation_rate=120.0,
        component_metadata=None,
    ):
        """
        Initializes the JaxSim class with a world, a system, and a simulation time step.

        Parameters:
        sim_obj : object
            The world context for the simulation.
        inputs : list
            The input parameters for the simulation.
        outputs : list
            The output parameters from the simulation.
        state : list
            The current state of the simulation.
        dictionary : dict
            A dictionary mapping component IDs to their names.
        entity_dict : dict
            A dictionary mapping entity names to their IDs.
        component_entity_dict : dict
            A dictionary mapping component names to a list of their entity IDs.
        map : list, optional
            An index map for ordering inputs and outputs.
        workers : int, optional
            The `World.run(workers=...)` count the systems were traced for.
        simulation_rate : float, optional
            Nominal model steps per second. The native caller still owns invocation timing.
        component_metadata : dict, optional
            Component metadata retained for native model introspection.
        """

        self.py_sim = sim_obj
        self.inputs = inputs
        self.outputs = outputs
        self.state = [np.array(a) for a in state]
        self.dictionary = dictionary
        self.entity_dict = entity_dict
        self.component_entity_dict = component_entity_dict
        self.workers = workers
        self.simulation_rate = simulation_rate
        self.component_metadata = component_metadata or {}
        if map is None:
            self.map = self.generate_index_map(self.inputs, self.outputs)
        else:
            self.map = map

        self._tick_steps = 0  # Track total steps taken

        # Find tick component location (it's always present when SimulationTick is used)
        self._tick_index = None
        for c_p, d_p, name in self.map:
            if name == "tick":
                self._tick_index = d_p
                break

    def generate_index_map(self, desired_order, current_order):
        """
        Generates an index map for reordering elements from current to desired order.

        Parameters:
        desired_order : list
            The desired order of elements.
        current_order : list
            The current order of elements.

        Returns:
        list
            A list of tuples containing current position (None when the element is missing
            from the current order), desired position, and element name.
        """
        index_map = []
        for d_p, element in enumerate(desired_order):
            c_p = current_order.index(element) if element in current_order else None
            index_map.append((c_p, d_p, self.dictionary[element]))
        return index_map

    def step_fn(self, state):
        """
        Advances a state by one tick, leaving `self.state` untouched.

        Parameters:
        state : list
            A state like `self.state`, one array per component.

        Returns:
        list
            The state after the tick. Components no system writes carry over unchanged.
        """
        state = list(state)
        if self._tick_index is not None:
            # Keep the tick's integer dtype; adding 1.0 would promote it to f64 and retrace py_sim
            state[self._tick_index] = state[self._tick_index] + 1
        out = self.py_sim(*state)
        return [state[d_p] if c_p is None else out[c_p] for c_p, d_p, _ in self.map]

    def export_model(self, path, interfaces=None, cpu="native"):
        """
        Write this passive simulation step and its native interface metadata.

        The directory holds `step.stablehlo.mlir`, one tick from state to next state as
        `step_fn` defines it; `step.o`, the same tick compiled ahead of time; `state.bin`,
        `self.state` as raw bytes in state order; `schema.json`, a versioned description of the
        model and optional named row interfaces; and `model-manifest.json`, hashes and target
        information for the exported artifacts.

        Parameters:
        path : str or os.PathLike
            The directory to write; created if missing.
        interfaces : dict[str, elodin.BatchHandle], optional
            Named data views for a native caller. They describe rows and access only; they do
            not prescribe callbacks, hook order, pacing or orchestration.
        cpu : str, optional
            `native` uses this CPU's features; `generic` uses the target architecture baseline.
        """
        import hashlib
        import importlib.metadata
        import json
        import os

        import numpy

        from elodin.stablehlo import compile_object, lower_to_stablehlo, object_target

        if cpu not in ("native", "generic"):
            raise ValueError("cpu must be 'native' or 'generic'")
        os.makedirs(path, exist_ok=True)
        mlir = lower_to_stablehlo(lambda *state: tuple(self.step_fn(state)), self.state)
        entity_names = {id: name for name, id in self.entity_dict.items()}
        components = []
        for (output, slot, name), value in zip(self.map, self.state):
            array = numpy.asarray(value)
            entity_ids = list(self.component_entity_dict[name])
            rows = len(entity_ids)
            element_shape = list(array.shape[1:] if rows > 1 else array.shape)
            components.append(
                {
                    "name": name,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "element_shape": element_shape,
                    "entity_axis": "leading" if rows > 1 else "none",
                    "entities": [entity_names[id] for id in entity_ids],
                    "entity_ids": entity_ids,
                    "rows": rows,
                    "row_bytes": array.nbytes // rows,
                    "size": array.nbytes,
                    "written_by_step": output is not None,
                    "metadata": dict(self.component_metadata.get(name, {})),
                }
            )

        native_interfaces = []
        component_by_name = {component["name"]: component for component in components}
        for interface_name, batch in (interfaces or {}).items():
            fields = []
            for field_name, handle in batch.fields.items():
                component = component_by_name.get(handle.name)
                if component is None:
                    raise ValueError(
                        f"interface {interface_name!r} field {field_name!r} has no "
                        f"component {handle.name!r} in this model"
                    )
                unknown = set(handle.entity_ids) - set(component["entity_ids"])
                if unknown:
                    raise ValueError(
                        f"interface {interface_name!r} field {field_name!r} has "
                        f"unknown entity ids {sorted(unknown)}"
                    )
                if handle.row_bytes != component["row_bytes"]:
                    raise ValueError(
                        f"interface {interface_name!r} field {field_name!r} row is "
                        f"{handle.row_bytes} bytes, not {component['row_bytes']}"
                    )
                fields.append(
                    {
                        "name": field_name,
                        "component": handle.name,
                        "access": handle.access,
                        "entity_ids": list(handle.entity_ids),
                        "row_bytes": handle.row_bytes,
                    }
                )
            row_counts = {len(field["entity_ids"]) for field in fields}
            if len(row_counts) > 1:
                raise ValueError(f"interface {interface_name!r} fields have different row counts")
            native_interfaces.append(
                {
                    "name": interface_name,
                    "rows": next(iter(row_counts), 0),
                    "fields": fields,
                }
            )

        time_step = 1.0 / self.simulation_rate
        schema = {
            "format": "elodin.model",
            "abi_version": 2,
            "endianness": "little",
            "workers": self.workers,
            "simulation": {
                "rate_hz": self.simulation_rate,
                "time_step_s": time_step,
                "tick_component": "tick" if self._tick_index is not None else None,
                "time_step_component": (
                    "simulation_time_step"
                    if any(component["name"] == "simulation_time_step" for component in components)
                    else None
                ),
                "tick_advanced_by_step": self._tick_index is not None,
            },
            "target": object_target(cpu),
            "components": components,
            "interfaces": native_interfaces,
        }
        files = {
            "step.stablehlo.mlir": mlir.encode(),
            "step.o": compile_object(mlir, "elodin_model_kernel", self.workers, cpu),
            "state.bin": b"".join(numpy.asarray(value).tobytes(order="C") for value in self.state),
            "schema.json": json.dumps(schema, indent=2).encode(),
        }
        for name, contents in files.items():
            with open(os.path.join(path, name), "wb") as handle:
                handle.write(contents)
        try:
            version = importlib.metadata.version("elodin")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        manifest = {
            "format": "elodin.model-export",
            "version": 1,
            "elodin_version": version,
            "target": schema["target"],
            "files": {
                name: {"bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
                for name, contents in files.items()
            },
        }
        with open(os.path.join(path, "model-manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2)

    def step(self, max_steps):
        """
        Advances the simulation by a specified number of steps.

        Parameters:
        max_steps : int
            The maximum number of steps to simulate.
        """
        for _ in range(max_steps):
            self.state = self.step_fn(self.state)
        if self._tick_index is not None:
            self._tick_steps += max_steps

    def get(self, state, component_name, entity_name=None):
        """
        Reads a component from a state.

        Parameters:
        state : list
            A state like `self.state`.
        component_name : str
            The name of the component to read.
        entity_name : str, optional
            The entity to read it for; without it, every entity's value.

        Returns:
        object
            The component value.
        """
        d_p = self._slot(component_name)
        index = None if entity_name is None else self._entity_index(component_name, entity_name)
        return state[d_p] if index is None else state[d_p][index]

    def set(self, state, component_name, entity_name, value):
        """
        Writes a component into a copy of a state.

        Parameters:
        state : list
            A state like `self.state`.
        component_name : str
            The name of the component to write.
        entity_name : str or None
            The entity to write it for; None writes every entity's value at once.
        value : object
            The new value, with the shape `get` returns.

        Returns:
        list
            The new state.
        """
        d_p = self._slot(component_name)
        index = None if entity_name is None else self._entity_index(component_name, entity_name)
        current = state[d_p] if index is None else state[d_p][index]
        value = np.asarray(value, dtype=current.dtype)
        if current.shape != value.shape:
            raise Exception(
                f"Value shape: {value.shape} does not match component: {component_name}, entity: {entity_name} state shape: {current.shape}"
            )
        state = list(state)
        state[d_p] = value if index is None else state[d_p].at[index].set(value)
        return state

    def get_state(self, component_name=None, entity_name=None):
        """
        Retrieves the state of a specific component by name.

        Parameters:
        component_name : str
            The name of the component to retrieve the state for.
        entity_name : str
            The name of the entity whose state is to be retrieved.
        Returns:
        object
            The state of the specified component for the specified entity.
        """
        if component_name is None:
            return self.state
        return self.get(self.state, component_name, entity_name)

    def set_state(self, component_name, entity_name, value):
        """
        Sets the state of a specific component by name.

        Parameters:
        component_name : str
            The name of the component to set the state for.
        entity_name : str or None
            The name of the entity whose state is to be set; None sets every entity's value.
        value : object
            The value to set the state to.
        """
        self.state = self.set(self.state, component_name, entity_name, value)

    def _slot(self, component_name):
        for _, d_p, name in self.map:
            if name == component_name:
                return d_p
        raise Exception(f"Component {component_name} not found in world")

    def _entity_index(self, component_name, entity_name):
        """Row of the entity in the component, or None when the component has no entity axis."""
        try:
            entity_id = self.entity_dict[entity_name]
        except KeyError:
            raise Exception(f"Entity {entity_name} not found in world")
        entity_ids = self.component_entity_dict[component_name]
        try:
            index = entity_ids.index(entity_id)
        except ValueError:
            raise Exception(f"Entity {entity_name} not found in component {component_name}")
        # A component held by a single entity has no entity axis.
        return None if len(entity_ids) == 1 else index

    def print_dictionary(self):
        """
        Prints a dictionary of component names to entity names in the world. and the shape of the state.
        """
        for component_name, entity_ids in self.component_entity_dict.items():
            entity_names_shapes = []
            for name, id in self.entity_dict.items():
                if id in entity_ids:
                    shape = np.shape(self.get_state(component_name, name))
                    entity_names_shapes.append(f"{name} (shape: {shape})")
            print(f"{component_name}: {', '.join(entity_names_shapes)}")

    def get_tick_count(self):
        """Get the total number of steps taken with auto-tick management.

        Returns:
            Number of steps taken since initialization or last reset
        """
        return self._tick_steps

    def sim_flatten(self):
        """Flatten the JaxSim object for JAX.

        Returns:
            A tuple of (flattened_state, auxiliary_data)
        """
        state = self.state

        aux_data = (
            self.py_sim,
            self.inputs,
            self.outputs,
            self.dictionary,
            self.entity_dict,
            self.component_entity_dict,
            self.map,
            self.workers,
            self.simulation_rate,
            self.component_metadata,
        )

        return (state, aux_data)

    def sim_unflatten(aux_data, cls):
        """Reconstruct a JaxSim from flattened state and auxiliary data.

        Args:
            aux_data: The auxiliary data from tree_flatten
            children: The children arrays from tree_flatten

        Returns:
            A new JaxSim instance
        """
        (
            py_sim,
            inputs,
            outputs,
            dictionary,
            entity_dict,
            component_entity_dict,
            map,
            workers,
            simulation_rate,
            component_metadata,
        ) = aux_data

        return JaxSim(
            py_sim,
            inputs,
            outputs,
            cls,
            dictionary,
            entity_dict,
            component_entity_dict,
            map,
            workers,
            simulation_rate,
            component_metadata,
        )


register_pytree_node(JaxSim, JaxSim.sim_flatten, JaxSim.sim_unflatten)
