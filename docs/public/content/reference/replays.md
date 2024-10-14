+++
title = "Replays"
description = "Replays"
draft = false
weight = 105
sort_by = "weight"
template = "reference/page.html"

[extra]
toc = true
top = false
icon = ""
order = 5
+++

Elodin has built-in functionality to save simulation data to a directory. This data can be used to replay the simulation at a later time. This is useful for debugging, testing, and sharing simulations. Replays also enable running simulations in a headless environment, and then replaying them in a graphical environment.

To replay a simulation using the Elodin editor, run the following command:

```bash
elodin editor <path to replay dir>
```

You can download an example replay archive from the following url: [replay.tar.gz](https://storage.googleapis.com/elodin-releases/docs/replay.tar.gz).


## File Layout

An Elodin replay directory contains:
- `metadata.json`
- `assets.bin`
- Component data Parquet files

### metadata.json

This file contains all of the metadata that is needed to replay the simulation. Its schema is as follows:

```json
{
  "sim_time_step": 0.008333333,
  "archetypes": {
    "archetype_a": [
      {
        "name": "component_1",
        "component_type": {
          "primitive_ty": "F64",
          "shape": [6]
        },
        "tags": null,
        "asset": false
      }
    ]
  }
}
```

There are two top-level keys in the metadata file:
- `"time_step"`: The time step that the simulation was run with.
- `"archetypes"`: A dictionary where the key is the name of the archetype, and the value is a list of component dictionaries.

Each component dictionary has the following keys:
- `"name"`: The name of the component.
- `"component_type"`: A dictionary containing the type information of the component.
  - `"primitive_ty"`: The primitive type of the component, which can be either "F64" or "U64".
  - `"shape"`: The shape is a list of integers that specifies the size of each dimension (e.g. `[]` for scalars, `[3]` for 3D vectors).

### assets.bin

This file is a container of non-component data that's associated with entities. See [Well-Known Assets] for some examples. Archetypes are often used to link entities to assets by using the asset handle as scalar component data. Some examples of such archetypes are `shape`, `asset_handle_panel`, and `asset_handle_entity_metadata`.

{% alert(kind="warning") %}
This file format is currently not suitable for consumption by external tools and is subject to change in the future.
{% end %}

### Component data Parquet files

Each [Archetype] in the simulation has a Parquet file named after it. The file contains the component data for each entity of that archetype. There is a row for each entity at every tick of the simulation. There are special columns in the Parquet file that are used to store the entity ID and the tick number: "entity_id" and "tick" respectively. The rest of the columns are the components of the archetype. E.g.:

```
│ world_pos                 ┆ world_vel                 ┆ tick ┆ entity_id │
│ array[f64, 7]             ┆ array[f64, 6]             ┆ u64  ┆ u64       │
│---------------------------┆---------------------------┆------┆-----------│
│ [0.0, 0.0, ... 5.999659]  ┆ [0.0, 0.0, ... -0.08175]  ┆ 0    ┆ 0         │
│ [0.0, 0.0, ... 5.9986375] ┆ [0.0, 0.0, ... -0.1635]   ┆ 1    ┆ 0         │
│ [0.0, 0.0, ... 5.996934]  ┆ [0.0, 0.0, ... -0.24525]  ┆ 2    ┆ 0         │
```

In this simplified example, there is a single entity with two components: `world_pos` and `world_vel`. The entity has an ID of `0`, and the simulation has run for three ticks.

## Well-Known Archetypes

### body

`body` is a core archetype that's used in most Elodin simulations. It represents the state of a rigid body with six degrees of freedom, and is used by the built-in 6DOF integrators to propagate the simulation forward in time. It also contains most of the necessary information to render the body in a graphical environment. The components of the `Body` archetype are:

- `world_pos`: A [SpatialTransform] representing the body's position and orientation in the world frame.
- `world_vel`: A [SpatialMotion] representing the body's linear and angular velocity in the world frame.
- `world_accel`: A [SpatialMotion] representing the body's linear and angular acceleration in the world frame.
- `force`: A [SpatialForce] representing the net force and torque acting on the body in the world frame.
- `inertia`: A [SpatialInertia] representing the body's mass, moment of inertia, and momentum in the body frame.

### shape

`shape` represents the primitive shape and color of a body. It serves as a link between entity ids and asset ids, as can be seen from its components:

- `asset_handle_material`: A `u64` handle to a [Material] asset.
- `asset_handle_mesh`: A `u64` handle to a [Mesh] asset.

### asset_handle_panel

`asset_handle_panel` links an entity to a panel asset. It only has one component: `asset_handle_panel`, which is a `u64` handle to a [Panel] asset.

### asset_handle_entity_metadata

`asset_handle_entity_metadata` links an entity to a metadata asset. It only has one component: `asset_handle_entity_metadata`, which is a `u64` handle to a metadata asset.

## Well-Known Assets

### Mesh

`Mesh` assets describe the geometry of a body. Currently, the mesh must be a primitive shape, such as a sphere, box, or cylinder.

### Material

`Material` assets describe the color and texture of a body. Currently, the material must be a solid color.

### GLB

`GLB` assets are 3D models in the GLB format. They can be used to represent complex geometries and textures.

### Panel

`Panel` assets describe the layout and contents of the Elodin GUI declaratively. They are used to configure viewports and graphs.

[Archetype]: /reference/python-api#archetypes
[SpatialTransform]: /reference/python-api#class-elodin-spatialtransform
[SpatialMotion]: /reference/python-api#class-elodin-spatialmotion
[SpatialForce]: /reference/python-api#class-elodin-spatialforce
[SpatialInertia]: /reference/python-api#class-elodin-spatialinertia

[Well-Known Assets]: #well-known-assets
[Mesh]: #mesh
[Material]: #material
[GLB]: #glb
