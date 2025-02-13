class JaxSim:
    """
    A class to simulate a system within a given world context.

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
        An index map for ordering inputs and outputs.
    """

    def __init__(
        self, sim_obj, inputs, outputs, state, dictionary, entity_dict, component_entity_dict
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
        """

        self.py_sim = sim_obj
        self.inputs = inputs
        self.outputs = outputs
        self.state = state
        self.dictionary = dictionary
        self.entity_dict = entity_dict
        self.component_entity_dict = component_entity_dict
        self.map = self.generate_index_map(self.inputs, self.outputs)

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
            A list of tuples containing current position, desired position, and element name.
        """
        index_map = []
        for element in desired_order:
            c_p = current_order.index(element)
            d_p = desired_order.index(element)
            name = self.dictionary[element]
            index_map.append((c_p, d_p, name))

        return index_map

    def order_array(self, array):
        """
        Orders an array according to the index map.

        Parameters:
        array : list
            The array to be ordered.

        Returns:
        list
            The ordered array.
        """
        ordered = [0] * len(self.map)
        for c_p, d_p, name in self.map:
            ordered[d_p] = array[c_p]
        return ordered

    def step(self, max_steps):
        """
        Advances the simulation by a specified number of steps.

        Parameters:
        max_steps : int
            The maximum number of steps to simulate.
        """
        for steps in range(max_steps):
            self.state = self.py_sim(*self.state)
            self.state = self.order_array(self.state)

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
        else:
            if entity_name is None:
                for c_p, d_p, name in self.map:
                    if name == component_name:
                        return self.state[d_p]
            else:
                entity_id = self.entity_dict[entity_name]
                component_entity_map = self.component_entity_dict[component_name]
                entity_index = component_entity_map.index(entity_id)

                for c_p, d_p, name in self.map:
                    if name == component_name:
                        return self.state[d_p][entity_index]

    def set_state(self, component_name, entity_name, value):
        """
        Sets the state of a specific component by name.

        Parameters:
        component_name : str
            The name of the component to set the state for.
        entity_name : str
            The name of the entity whose state is to be set.
        value : object
            The value to set the state to.

        Returns:
        object
            The state of the specified component for the specified entity.
        """
        if component_name is None:
            raise Exception("Component name must be provided")
        else:
            if entity_name is None:
                raise Exception("Entity name must be provided")
            else:
                try:
                    entity_id = self.entity_dict[entity_name]
                except KeyError:
                    raise Exception(f"Entity {entity_name} not found in world")
                try:
                    component_entity_map = self.component_entity_dict[component_name]
                except KeyError:
                    raise Exception(f"Component {component_name} not found in world")
                try:
                    entity_index = component_entity_map.index(entity_id)
                except ValueError:
                    raise Exception(f"Entity {entity_name} not found in component {component_name}")
                for c_p, d_p, name in self.map:
                    if name == component_name:
                        if self.state[d_p][entity_index].shape == value.shape:
                            self.state[d_p][entity_index] = value
                        else:
                            raise Exception(
                                f"Value shape: {value.shape} does not match component: {component_name}, entity: {entity_name} state shape: {self.state[d_p][entity_index].shape}"
                            )

    def print_dictionary(self):
        """
        Prints a dictionary of component names to entity names in the world. and the shape of the state.
        """
        for component_name, entity_ids in self.component_entity_dict.items():
            entity_names_shapes = []
            for name, id in self.entity_dict.items():
                if id in entity_ids:
                    # Find the corresponding state shape
                    for c_p, d_p, comp_name in self.map:
                        if comp_name == component_name:
                            entity_index = self.component_entity_dict[component_name].index(id)
                            shape = self.state[d_p][entity_index].shape
                            entity_names_shapes.append(f"{name} (shape: {shape})")
            print(f"{component_name}: {', '.join(entity_names_shapes)}")
