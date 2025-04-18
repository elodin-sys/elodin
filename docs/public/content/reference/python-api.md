+++
title = "Python API"
description = "Python API"
draft = false
weight = 103
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 3
+++

## World

An Elodin simulation begins with a `World` object. The `World` object is the root of the simulation hierarchy and provides methods for composing
and running the simulation. The `World` object also provides helper methods for displaying entities and graphs in the editor.

### _class_ `elodin.World`

The Elodin simulation world.
- `__init__()` -> [elodin.World]

    Create a new world object.

- `spawn(archetypes, name)` -> [elodin.EntityId]

    Spawn a new entity with the given archetypes and name.
    - `archetypes` : one or many [Archetypes],
    - `name` : optional name of the entity

- `insert(id, archetypes)` -> None

    Insert archetypes into an existing entity.
    - `id` : [elodin.EntityId], the id of the entity to insert into.
    - `archetypes` : one or many [Archetypes]

- `insert_asset(asset)` -> handle reference

    Insert a 3D asset into the world.
    - `asset` [elodin.Mesh] | [elodin.Material] : the asset to insert, allows for loading the mesh once and using it in multiple shapes.

- `shape(mesh, material)` -> [elodin.Shape]

    Create a shape as an Elodin Shape Archetype.
    - `mesh`: the mesh of the shape,
    - `material`: the material of the shape

- `glb(url)` -> [elodin.Scene]

    Load a GLB asset as an Elodin Scene Archetype.
    - `url`: the URL or filepath of the GLB asset

- `run(system, sim_time_step, run_time_step, default_playback_speed, max_ticks, optimize)` -> None

    Run the simulation.
    - `system` : [elodin.System], the systems to run, can be supplied as a list of systems delineated by pipes.
    - `sim_time_step` : `float`, optional, the amount of simulated time between each tick, defaults to 1 / 120.0.
    - `run_time_step` : `float`, optional, the amount of real time between each tick, defaults to real-time playback by matching the `sim_time_step`.
    - `default_playback_speed` : `float`, optional, the default playback speed of the Elodin client when running this simulation, defaults to 1.0 (real-time).
    - `max_ticks` : `integer`, optional, the maximum number of ticks to run the simulation for before stopping.
    - `optimize` : `bool`, optional flag to enable runtime optimizations for the simulation code, defaults to `False`. If optimizations are enabled, the simulation will start slower but run faster.

### _class_ `elodin.EntityId`
Integer reference identifier for entities in Elodin.

### _class_ `elodin.Panel`
A configuration object for creating a panel view in the Elodin Client UI.

- `Panel.viewport(track_entity, track_rotation, fov, active, pos, looking_at, show_grid, hdr, name)` -> [elodin.Panel]

    Create a viewport panel.

    - `track_entity` : [elodin.EntityId], optional, the entity to track.
    - `track_rotation` : `boolean`, whether to track the rotation of the entity, defaults to `True`.
    - `fov` : `float`, the field of view of the camera, defaults to `45.0`.
    - `active` : `boolean`, whether the panel is active, defaults to `False`.
    - `pos` : `list`, optional, the position of the camera.
    - `looking_at` : `list`, optional, the point the camera is looking at.
    - `show_grid` : `boolean`, whether to show the grid, defaults to `False`.
    - `hdr` : `boolean`, whether to use HDR rendering, defaults to `False`.
    - `name` : `string`, optional, the name of the panel.

- `Panel.graph(*entities, name)` -> [elodin.Panel]

    Create a graph panel.

    - `*entities` : Sequence of [elodin.GraphEntity] objects to include in the graph.
    - `name` : `string`, optional, the name of the panel.

- `Panel.vsplit(*panels, active)` -> [elodin.Panel]

    Create a vertical split panel.

    - `*panels` : Sequence of [elodin.Panel] objects to vertically split across.
    - `active` : `boolean`, whether the panel is active, defaults to `False`.

- `Panel.hsplit(*panels, active)` -> [elodin.Panel]

    Create a horizontal split panel.

    - `*panels` : Sequence of [elodin.Panel] objects to horizontally split across.
    - `active` : `boolean`, whether the panel is active, defaults to `False`.

### _class_ `elodin.GraphEntity`

A configuration object for creating a graph entity in the Elodin Client UI.

- `__init__(entity_id, *components)` -> [elodin.GraphEntity]

    Create a graph entity.

    - `entity_id` : [elodin.EntityId], the entity to graph.
    - `*components` : Sequence of `elodin.ShapeIndexer` indexes of components to graph.

###  _class_ `elodin.Mesh`

A built in class for creating basic 3D meshes.

- `Mesh.cuboid(x: float, y: float, z: float)` -> [elodin.Mesh]

    Create a cuboid mesh with dimensions `x`, `y`, and `z`.

- `Mesh.sphere(radius: float)` -> [elodin.Mesh]

    Create a sphere mesh with radius `radius`.

###  _class_ `elodin.Material`

A built in class for creating basic 3D materials.

- `Material.color(r: float, g: float, b: float)` -> [elodin.Material]

    Create a material with RGB color values.

### _class_ `elodin.Shape`

`Shape` describes a basic entity for rendering 3D assets in Elodin.

- `__init__(mesh, material)` -> [elodin.Shape]

  Create a shape archetype initialized to the provided mesh and material.

  - `mesh` : handle reference returned from `World.insert_asset()` using the [elodin.Mesh] class.
  - `material` : handle reference returned from `World.insert_asset()` using the [elodin.Material] class.

### _class_ `elodin.Scene`

`Scene` describes a complex scene entity loaded from a glb file.

- `__init__(glb)` -> [elodin.Scene]

  Create a scene from a loaded file.

  - `glb` : handle reference returned from `World.insert_asset()` using the `elodin.Glb` class.


#### Example

This example creates a simple simulation with a spinning cuboid body:

```python
import elodin as el
import jax.numpy as jnp

@el.map
def spin(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.Force(torque=(inertia.mass() * jnp.array([0.0, 1.0, 0.0])))

w = el.World()

mesh = w.insert_asset(el.Mesh.cuboid(0.1, 0.8, 0.3))
material = w.insert_asset(el.Material.color(25.3, 18.4, 1.0))

cuboid_id = w.spawn([el.Body(), el.Shape(mesh, material)], name="cuboid")

camera = el.Panel.viewport(pos=[0.0, -5.0, 0.0], hdr=True, name="camera")
graph = el.Panel.graph(
    el.GraphEntity(cuboid_id, *el.Component.index(el.WorldPos)[:4]), name="graph"
)

w.spawn(el.Panel.vsplit(camera, graph), name="main_view")

sys = el.six_dof(sys=spin)
sim = w.run(sys, sim_time_step=1.0 / 120.0)
```

<br></br>
## 6 Degrees of Freedom Model

Elodin has a built-in [6 Degrees of Freedom](https://en.wikipedia.org/wiki/Six_degrees_of_freedom) (6DoF) system
implementation for simulating [rigid bodies](https://en.m.wikipedia.org/wiki/Rigid_body), such as flight vehicles.
You can review the implementation [here](https://github.com/elodin-sys/elodin/blob/332957c41f609e1ccee36dbc48750ea59001c817/libs/nox-ecs/src/six_dof.rs).
Using the associated [elodin.Body] archetype and prebuilt components, we can create a 6DoF system that aligns closely with this
[familiar model](https://www.mathworks.com/help/aeroblks/6dofquaternion.html) from Simulink.

### _function_ `elodin.six_dof`
- `six_dof(time_step, sys, integrator)` -> [elodin.System]

    Create a system that models the 6DoF dynamics of a rigid body in 3D space. The provided set of systems can be integrated
    as effectors using the provided `integrator` and simulated in a world with a given `time_step`.

    - `time_step` : `float`, The time step used when integrating a body's acceleration into its velocity and position. Defaults
    to the `sim_time_step` provided in World.run(...) if unset
    - `sys` : one or more [elodin.System] instances used as effectors
    - `integrator` : [elodin.Integrator], default is `Integrator.Rk4`

### _class_ `elodin.Integrator`

- `elodin.Integrator.Rk4` -> [elodin.Integrator]

    Runge-Kutta 4th Order (RK4) Integrator: Elodin provides a built-in implementation for a [4th order Runge-Kutta integrator](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods).
    The RK4 integrator is a numerical method used to solve ordinary differential equations. You can review the implementation [here](https://github.com/elodin-sys/elodin/blob/332957c41f609e1ccee36dbc48750ea59001c817/libs/nox-ecs/src/integrator/rk4.rs).

- `elodin.Integrator.SemiImplicit` -> [elodin.Integrator]

    Semi-Implicit Integrator: Elodin provides a built-in implementation for a [semi-implicit Euler integrator](https://en.wikipedia.org/wiki/Semi-implicit_Euler_method).
    The semi-implicit integrator is a numerical method used to solve ordinary differential equations. You can review the implementation [here](https://github.com/elodin-sys/elodin/blob/332957c41f609e1ccee36dbc48750ea59001c817/libs/nox-ecs/src/integrator/semi_implicit.rs).

### _class_ `elodin.Body`

`Body` is an archetype that represents the state of a rigid body with six degrees of freedom. It provides all of the spatial information necessary for the [elodin.six_dof] system

- `__init__(world_pos, world_vel, inertia, force, world_accel)` -> [elodin.Body]

  Create a body archetype initialized to the provided values.

  - `world_pos` : [elodin.WorldPos], default is SpatialTransform()
  - `world_vel` : [elodin.WorldVel], default is SpatialMotion()
  - `inertia` : [elodin.Inertia], default is SpatialInertia(1.0)
  - `force` : [elodin.Force], default is SpatialForce()
  - `world_accel` : [elodin.WorldAccel], default is SpatialMotion()

  {% alert(kind="warning") %}
  Inertia is in body frame, all other representations are in the world frame.
  {% end %}

    #### _class_ `elodin.WorldPos`

    `WorldPos` is an annotated [elodin.SpatialTransform] component that represents the world frame position of a body in 3D space. See [elodin.SpatialTransform] for usage.

    #### _class_ `elodin.WorldVel`

    `WorldVel` is an annotated [elodin.SpatialMotion] component that represents the world frame velocity of a body in 3D space. See [elodin.SpatialMotion] for usage.

    #### _class_ `elodin.Inertia`

    `Inertia` is an annotated [elodin.SpatialInertia] component that represents the body frame inertia of a body in 3D space. See [elodin.SpatialInertia] for usage.

    #### _class_ `elodin.Force`

    `Force` is an annotated [elodin.SpatialForce] component that represents the world frame forces applied to a body in 3D space. See [elodin.SpatialForce] for usage.

    #### _class_ `elodin.WorldAccel`

    `WorldAccel` is an annotated [elodin.SpatialMotion] component that represents the world frame acceleration of a body. See [elodin.SpatialMotion] for usage.

#### Example
A simple example of a 6DoF system that models gravity acting on a rigid body in 3D space.

```python
import elodin as el
import jax.numpy as jnp

SIM_TIME_STEP = 1.0 / 120.0

@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.Force(linear=(inertia.mass() * jnp.array([0.0, -9.81, 0.0])))

w = el.World()
w.spawn(el.Body(), name="example")
sys = el.six_dof(sys=gravity, integrator=el.Integrator.Rk4)
sim = w.run(sys, SIM_TIME_STEP)
```

{% alert(kind="warning") %}
You should never need to use the six_dof time_step parameter unless you need to simulate a sensor at a specific frequency different from the world simulation. This is an advanced feature and should be used with caution, and likely a symptom of needing to move your testing into your flight software & communicate with the simulation over Impel.
{% end %}
```python
# lower frequency time step
SIX_DOF_TIME_STEP = 1.0 / 60.0
sys = el.six_dof(time_step=SIX_DOF_TIME_STEP, sys=gravity)
sim = w.run(sys, SIM_TIME_STEP)
```

<br></br>
## Components

Components are containers of data that is associated with an entity. See [ECS Data Model](/reference/overview#ecs-data-model) for more context on entities and components.

To define a new component, add [elodin.Component] as metadata to a base class using [typing.Annotated]. The base class can be [jax.Array] or some other container of array data. This is an example of a component that annotates [jax.Array]:

```python
import elodin as el

Wind = typing.Annotated[
    jax.Array,
    el.Component(
        "wind",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]
```

### _class_ `elodin.Component`

A container of component metadata.

- `__init__(name, type = None, asset = False, metadata = {})` -> [elodin.Component]

    Create a new component with:
    - Unique name (e.g. "world_pos, "inertia").
    - Component type information (via [elodin.ComponentType]). This is optional if the base class already provides component type information as part of `__metadata__`, which is the case for [elodin.Quaternion], [elodin.Edge], and all [spatial vector algebra](#spatial-vector-algebra) classes.
    - Flag indicating whether the component is an asset (e.g. a mesh, texture, etc.).
    - Other metadata that is optional (e.g. description, units, labels, etc.).

    ```python
    import elodin as el

    el.Component(
        "wind",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
    ```

    The above example defines a "wind" component that is a 3D vector of `float64` values. The "element_names" entry is an example of optional metadata. It specifies the labels for each element of the vector that are displayed in the component inspector.

- `Component.name(component)` -> `string`

    The unique name of the component.

- `Component.index(component)` -> `elodin.ShapeIndexer`

    A shape indexer that can be used to access the component data.


### _class_ `elodin.ComponentType`

`ComponentType` describes the shape and data type of a component. The shape is a tuple of integers that specifies the size of each dimension (e.g. `()` for scalars, `(3,)` for 3D vectors). The data type is an [elodin.PrimitiveType].

- `__init__(dtype, shape)` -> [elodin.ComponentType]

    Create a component type from a data type and shape.

### _class_ `elodin.Edge`

An edge is a relationship between two entities. See [elodin.GraphQuery] for information on how to use edges in graph queries.

- `__init__(left, right)` -> [elodin.Edge]

    Create an edge between two entities given their unique ids.

<br></br>
## Archetypes

An archetype is a combination of components with a unique name. To define a new archetype, create a subclass of `elodin.Archetype` with the desired components as fields. Here is an example of an archetype for a kalman filter:

{% alert(kind="notice") %}
To automatically generate `__init__()`, you can use the [@dataclass decorator.](https://docs.python.org/3/library/dataclasses.html)
{% end %}

```python
import elodin as el
from dataclasses import dataclass

@dataclass
class KalmanFilter(el.Archetype):
    p: P
    att_est: AttEst
    ang_vel_est: AngVelEst
    bias_est: BiasEst
```

The archetype can then be used to attach components to entities:

```python
world.insert(
    satellite,
    KalmanFilter(
        p=np.identity(6),
        att_est=el.Quaternion.identity(),
        ang_vel_est=np.zeros(3),
        bias_est=np.zeros(3),
    ),
)
```

<br></br>
## Systems

Systems are the building blocks of simulation; they are functions that operate on a set of input components and produce a set of output components. Elodin provides decorators that allow for systems to be easily defined from functions.

### `@elodin.system`

{% alert(kind="notice") %}
This is a lower-level primitive; for many cases `@elodin.map` – a wrapper around `@elodin.system` – is easier to use.
{% end %}

This is a lower-level API for defining a system. A function decorated with `@elodin.system` accepts special parameter types (such as [elodin.Query] and [elodin.GraphQuery]) that specify what data the system needs access to. It returns an [elodin.Query] containing one or more [components]. Some examples of `@elodin.system` are:

```python
import elodin as el

@el.system
def gravity(
    graph: el.GraphQuery[GravityEdge],
    query: el.Query[el.WorldPos, el.Inertia],
) -> el.Query[el.Force]: ...

@el.system
def apply_wind(
    w: el.Query[Wind], q: el.Query[el.Force, el.WorldVel]
) -> el.Query[el.Force]: ...
```

### `@elodin.map`

{% alert(kind="notice") %}
Graph queries cannot be used with `@elodin.map`. Use `@elodin.system` instead.
{% end %}

This is a higher-level API for defining a system that reduces the boilerplate of `@elodin.system` by unpacking the input and output queries into individual components, and wrapping the body of the function in a `query.map(ret_type, ...)` call. It is useful for systems with simple data flow patterns. Some examples of `@elodin.map` are:

```python
import elodin as el

@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force: ...

@el.map
def gyro_omega(vel: el.WorldVel) -> GyroOmega: ...
```

The following systems are equivalent as the `@elodin.map` definition effectively desugars to the `@elodin.system` one:

```python
import elodin as el

@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=inertia.mass() * jnp.array([0.0, -9.81, 0.0]))

@el.system
def gravity(query: el.Query[el.Force, el.Inertia]) -> el.Query[el.Force]:
    return query.map(
        el.Force,
        lambda f, inertia: f + el.SpatialForce(linear=inertia.mass() * jnp.array([0.0, -9.81, 0.0])),
    )
```

### _class_ `elodin.Query`

`Query` is the primary mechanism for accessing data in Elodin. It is a view into the world state that is filtered by the components specified in the query. Only entities that have been spawned with all of the query's components will be selected for processing. For example, the query `Query[WorldPos, Inertia]` would only select entities that have both a `WorldPos` and an `Inertia` component (typically via the `Body` [archetype](#archetypes)).

- `map(ret_type, map_fn)` -> [elodin.Query]

    Apply a function `map_fn` to the query's components and return a new query with the specified `ret_type` return type. `map_fn` should be a function that takes the query's components as arguments and returns a single value of type `ret_type`.

    ```python
    import elodin as el

    @el.system
    def gravity(query: el.Query[el.Force, el.Inertia]) -> el.Query[el.Force]:
        return query.map(
            el.Force,
            lambda f, inertia: f + el.SpatialForce(linear=inertia.mass() * jnp.array([0.0, -9.81, 0.0])),
        )
    ```

    In this example, `ret_type` is `el.Force` and `map_fn` is a lambda function with the signature `(el.Force, el.Inertia) -> el.Force`.


    To return multiple components as output, `ret_type` must be a tuple:
    ```python
    import elodin as el

    @el.system
    def multi_out_sys(query: el.Query[A]) -> el.Query[C, D]:
        return query.map((C, D), lambda a: ...)
    ```

### _class_ `elodin.GraphQuery`

`GraphQuery` is a special type of query for operating on edges in an entity graph. [Edges](#class-elodin-edge) represent relationships between entities and are fundamental for modeling physics systems such as gravity.

A `GraphQuery` requires exactly one type argument, which must be an annotated [elodin.Edge] component. For example, `GraphQuery[GravityEdge]` is a valid graph query if `GravityEdge` is a component with `Edge` as the base class:

```python
GravityEdge = typing.Annotated[elodin.Edge, elodin.Component("gravity_edge")]
```

- `edge_fold(left_query, right_query, return_type, init_value, fold_fn)` -> [elodin.Query]

    For each edge, query the left and right entity components using `left_query` and `right_query`, respectively. Then, apply the `fold_fn` function to those input components to compute the `return_type` output component(s).

    {% alert(kind="notice") %}
    The `return_type` component(s) must belong to the **left** entity of the edge.
    {% end %}

    A single left entity may have edges to multiple right entities, but it can only hold a single value for each `return_type` component. So, the `fold_fn` computations for each entity's edges must be accumulated into a single final value. To carry the intermediate results, `fold_fn` takes an "accumulator" value as the first argument. Its output is set as the accumulator value for the next iteration. `init_value` is the initial value of the accumulator.

    {% alert(kind="notice") %}
    `edge_fold` makes no guarantees about the order in which edges are processed. For associative operators like `+`, the order the elements are combined in is not important, but for non-associative operators like `-`, the order will affect the final result.
    {% end %}

    See the [Three-Body Orbit Tutorial](/home/3-body) for a practical example of using `edge_fold` to compute gravitational forces between entities.

<br></br>
## Primitives

### _class_ `elodin.PrimitiveType`

- `elodin.PrimitiveType.F64` -> [elodin.PrimitiveType]

    A constant representing the 64-bit floating point data type.

- `elodin.PrimitiveType.U64` -> [elodin.PrimitiveType]

    A constant representing the 64-bit unsigned integer data type.

### _class_ `elodin.Quaternion`

Unit quaternions are used to represent spatial orientations and rotations of bodies in 3D space.

- `Quaternion.identity()` -> [elodin.Quaternion]

    Create a unit quaternion with no rotation.

- `Quaternion.from_axis_angle()` -> [elodin.Quaternion]

    Create a quaternion from an axis and an angle.

- `inverse()` -> [elodin.Quaternion]

    Compute the inverse of the quaternion.

- `normalize()` -> [elodin.Quaternion]

    Normalize to a unit quaternion.

- `integrate_body(body_delta)` -> [elodin.Quaternion]

    Perform quaternion integration in body-frame with angular velocity `body_delta`, which must be a 3D vector.

- `__add__(other)` -> [elodin.Quaternion]

    Add two quaternions.

    {% alert(kind="info") %}
    Adding quaternions does not yield the composite rotation unless they are [infinitesimal rotations, use multiplication instead.](https://en.wikipedia.org/wiki/Rotation_matrix#Infinitesimal_rotations)
    {% end %}


- `__mul__(other)` -> [elodin.Quaternion]

    Multiply two quaternions.

- `__matmul__(vector)` -> [jax.Array] | [elodin.SpatialTransform] | [elodin.SpatialMotion] | [elodin.SpatialForce]

    Rotate `vector` by computing the matrix product. The vector can be a plain [jax.Array] or one of the following spatial objects: [elodin.SpatialTransform], [elodin.SpatialMotion], [elodin.SpatialForce]. The return type is the same as the input type.

<br></br>
## Spatial Vector Algebra

Elodin uses Featherstone’s spatial vector algebra notation for rigid-body dynamics as it is a compact way of representing the state of a rigid body with six degrees of freedom. You can read a short into [here](https://homes.cs.washington.edu/~todorov/courses/amath533/FeatherstoneSlides.pdf) or in [Rigid Body Dynamics Algorithms (Featherstone - 2008)](https://link.springer.com/book/10.1007/978-1-4899-7560-7).

### _class_ `elodin.SpatialTransform`

A spatial transform is a 7D vector that represents a rigid body transformation in 3D space.

- `__init__(arr, angular, linear)` -> [elodin.SpatialTransform]

    Create a spatial transform from either `arr` or `angular` **and** `linear`. If no arguments are provided, the spatial transform is initialized to the default values of the identity quaternion and the zero vector.

    - `arr` : [jax.Array] with shape (7)
    - `angular` : [elodin.Quaternion], default is `Quaternion.identity()`
    - `linear` : [jax.Array] with shape (3), default is `[0, 0, 0]`

- `linear()` -> [jax.Array]

    Get the linear part of the spatial transform as a vector with shape (3,).

- `angular()` -> [elodin.Quaternion]

    Get the angular part of the spatial transform as a quaternion.

- `__add__(other)` -> [elodin.SpatialTransform] | [elodin.SpatialMotion]

    Add either a [elodin.SpatialTransform] or a [elodin.SpatialMotion] to the spatial transform. The return type is always a spatial transform.

### _class_ `elodin.SpatialMotion`

A spatial motion is a 6D vector that represents either the velocity or acceleration of a rigid body in 3D space.

- `__init__(angular, linear)` -> [elodin.SpatialMotion]

    Create a spatial motion from an angular and a linear vector. Both arguments are optional and default to zero vectors.

    - `angular` : [jax.Array] with shape (3), default is `[0, 0, 0]`
    - `linear` : [jax.Array] with shape (3), default is `[0, 0, 0]`

- `linear()` -> [jax.Array]

    Get the linear part of the spatial motion as a vector with shape (3,).

- `angular()` -> [jax.Array]

    Get the angular part of the spatial motion as a vector with shape (3,).

- `__add__(other)` -> [elodin.SpatialMotion]

    Add two spatial motions.

### _class_ `elodin.SpatialForce`

A spatial force is a 6D vector that represents the linear force and torque applied to a rigid body in 3D space.

- `__init__(arr, torque, force)` -> [elodin.SpatialForce]

    Create a spatial force from either `arr` or `torque` **and** `force`. If no arguments are provided, the spatial force is initialized to zero torque and force.

    - `arr` : [jax.Array] with shape (6)
    - `torque` : [jax.Array] with shape (3), default is `[0, 0, 0]`
    - `force` : [jax.Array] with shape (3), default is `[0, 0, 0]`

- `force()` -> [jax.Array]

    Get the linear force part of the spatial force as a vector with shape (3,).

- `torque()` -> [jax.Array]

    Get the torque part of the spatial force as a vector with shape (3,).

- `__add__(other)` -> [elodin.SpatialForce]

    Add two spatial forces.

### _class_ `elodin.SpatialInertia`

A spatial inertia is a 7D vector that represents the mass, moment of inertia, and momentum of a rigid body in 3D space. The moment of inertia is represented in its [diagonalized form] of $[I_1, I_2, I_3]$.

[diagonalized form]: https://en.wikipedia.org/wiki/Moment_of_inertia#Principal_axes

- `__init__(mass, inertia)` -> [elodin.SpatialInertia]

    Create a spatial tensor inertia from a scalar mass and an optional inertia tensor diagonal with shape (3,). If the inertia tensor is not provided, it is set to the same value as the mass along all axes.

- `mass()` -> [jax.Array]

    Get the scalar mass of the spatial inertia.

- `inertia_diag()` -> [jax.Array]

    Get the inertia tensor diagonal of the spatial inertia with shape (3,).

[elodin.World]: #class-elodin-world
[elodin.EntityId]: #class-elodin-entityid
[elodin.Panel]: #class-elodin-panel
[elodin.GraphEntity]: #class-elodin-graphentity
[elodin.Mesh]: #class-elodin-mesh
[elodin.Material]: #class-elodin-material
[elodin.Shape]: #class-elodin-shape
[elodin.Scene]: #class-elodin-scene

[elodin.six_dof]: #function-elodin-six-dof
[elodin.Integrator]: #class-elodin-integrator
[elodin.Body]: #class-elodin-body
[elodin.WorldPos]: #class-elodin-worldpos
[elodin.WorldVel]: #class-elodin-worldvel
[elodin.Inertia]: #class-elodin-inertia
[elodin.Force]: #class-elodin-force
[elodin.WorldAccel]: #class-elodin-worldaccel

[Components]: #components
[elodin.Component]: #class-elodin-component
[elodin.ComponentType]: #class-elodin-componenttype
[elodin.Edge]: #class-elodin-edge
[typing.Annotated]: https://docs.python.org/3/library/typing.html#typing.Annotated

[Archetypes]: #archetypes

[elodin.System]: #elodin-system
[elodin.Query]: #class-elodin-query
[elodin.GraphQuery]: #class-elodin-graphquery

[elodin.PrimitiveType]: #class-elodin-primitivetype
[elodin.Quaternion]: #class-elodin-quaternion

[elodin.SpatialTransform]: #class-elodin-spatialtransform
[elodin.SpatialMotion]: #class-elodin-spatialmotion
[elodin.SpatialForce]: #class-elodin-spatialforce
[elodin.SpatialInertia]: #class-elodin-spatialinertia
[jax.Array]: https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html#jax.Array
[jax.typing.ArrayLike]: https://jax.readthedocs.io/en/latest/_autosummary/jax.typing.ArrayLike.html#jax.typing.ArrayLike
