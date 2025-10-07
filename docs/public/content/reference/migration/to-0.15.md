+++
title = "v0.14 to v0.15"
description = "Elodin Migration to v0.15"
draft = false
weight = 104
sort_by = "weight"

[extra]
lead = "Migration guide for Elodin v0.14 to v0.15"
toc = true
top = false
order = 7
icon = ""
+++

Elodin v0.15.0 is a breaking change that introduces the schematic configuration. This document shows how to upgrade code using Elodin v0.14.*.

## Broad Strokes

The specification of UI panels, e.g., viewports, graphs, etc., and the UI representation of objects has been changed from Python to a KDL format called "schematics". This permits changes in the editor to be saved and reloaded without manipulating the Python code. For this section, the 0.14.2 ball example will be converted. The next section will show how to convert individual elements.

### Remove Asset Handling

The ball example had the following code from v0.14.2:

```python
ball_mesh = world.insert_asset(el.Mesh.sphere(BALL_RADIUS))
ball_color = world.insert_asset(el.Material.color(0.1, 0.2, 1.0))
world.spawn(
    [
        el.Body(world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 6.0]))),
        el.Shape(ball_mesh, ball_color),
        WindData(seed=jnp.int64(seed)),
    ],
    name="ball",
)
```
We can remove the asset handling.
```diff
-ball_mesh = world.insert_asset(el.Mesh.sphere(BALL_RADIUS))
-ball_color = world.insert_asset(el.Material.color(0.1, 0.2, 1.0))
world.spawn(
    [
        el.Body(world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 6.0]))),
-        el.Shape(ball_mesh, ball_color),
        WindData(seed=jnp.int64(seed)),
    ],
    name="ball",
)
```
### Remove UI Panel Spawning
Code that spawns UI elements can be removed.
```diff
-   world.spawn(
-       el.Panel.viewport(
-           active=True,
-           pos="(0.0,0.0,0.0, 0.0, 8.0, 2.0, 4.0)",
-           look_at="(0.0,0.0,0.0,0.0, 0.0, 0.0, 3.0)",
-           show_grid=True,
-           hdr=True,
-       ),
-       name="Viewport",
-   )
-   world.spawn(el.Line3d("ball.world_pos", line_width=2.0))
```
### Specify UI with New Schematics Call

All the preceding information can now be specified via schematics.

```python
    world.schematic("""
        hsplit {
            viewport name=Viewport pos="(0,0,0,1, 8,2,4)" look_at="(0,0,0,1, 0,0,3)" hdr=#true show_grid=#true active=#true
        }
        object_3d ball.world_pos {
            sphere radius=0.2 {
                color 25 50 255 128
            }
        }
        line_3d ball.world_pos line_width=2.0 {
            color yolk
        }
    """)
```
## Details
The old code is python, and the new code is specified in a [KDL](https://docs.rs/kdl/latest/kdl/) string to the `world.schematic()` method.


### `el.Material.color` to `color`
A color can be specified by its red, green, blue, and optionally its alpha components as integers from 0 (fully transparent) to 255 (fully opaque).

OLD
```python
el.Material.color(0.1, 0.2, 1.0))
```
NEW
```python
color 25 50 255 128
```

Omit the fourth value if you want the color to remain fully opaque. The same syntax works with named colors:

```
    graph "value" {
        color yolk 120
    }
```

Here `yolk` is the base color and `120` sets a partially transparent alpha.

It can also be specified by a few names:

* black
* white
* turquoise
* slate
* pumpkin
* yolk
* peach
* reddish
* hyperblue
* mint

NEW
```python
color mint
```

It is often specified as a child of what its describing, e.g., this sphere is reddish.
```
    sphere radius=0.2 {
        color reddish
    }
```

### `object_3d` ellipsoid
An `ellipsoid` block renders an ellipsoidal volume aligned with an entity's `world_pos`. Provide a scale expression that evaluates to three components (literal values or an EQL expression) and an optional translucent color.

```
    object_3d drone.world_pos {
        ellipsoid scale="(0.75, 0.35, 0.95)" {
            color 255 255 0 120
        }
    }
```

Because the scale expression is re-evaluated every frame, you can reference live data such as `rocket.ellipsoid_scale` to animate the ellipsoid.

### `el.Panel.hsplit` to `hsplit`
OLD
```python
el.Panel.hsplit(
    el.Panel.graph("ball.world_pos.x"),
    el.Panel.graph("ball.world_pos.y"),
)
```
NEW
```python
hsplit {
    graph ball.world_pos.x
    graph ball.world_pos.y
}
```

### `el.Panel.vsplit` to `vsplit`
OLD
```python
el.Panel.vsplit(
    el.Panel.graph("ball.world_pos.x"),
    el.Panel.graph("ball.world_pos.y"),
)
```
NEW
```python
vsplit {
    graph ball.world_pos.x
    graph ball.world_pos.y
}
```
### `el.Panel.tabs` to `tabs`
OLD
```python
el.Panel.tabs(
    el.Panel.graph("ball.world_pos.x"),
    el.Panel.graph("ball.world_pos.y"),
)
```
NEW
```python
tabs {
    graph ball.world_pos.x
    graph ball.world_pos.y
}
```
### `el.Panel.viewport` to `viewport`
OLD
```python
el.Panel.viewport(
    active=True,
    pos="(0.0,0.0,0.0, 0.0, 8.0, 2.0, 4.0)",
    look_at="ball.world_pos + (0.0,0.0,0.0,0.0, 0.0, 0.0, 3.0)",
    show_grid=True,
    hdr=True,
)
```
NEW
```python
viewport name=Viewport pos="(0,0,0,0, 8,2,4)" look_at="ball.world_pos + (0,0,0,0, 0,0,3)" hdr=#true show_grid=#true active=#true
```

### `el.Panel.graph` to `graph`
OLD
```python
el.Panel.graph("ball.world_pos.x, ball.world_pos.y")
```
NEW
```python
graph "ball.world_pos, ball.world_pos.y" Name="ball position"
```
### `el.Line3d` to `line_3d`
OLD
```python
el.Line3d("ball.world_pos", line_width=2.0)
```
NEW
```kdl
line_3d ball.world_pos line_width=2.0 perspective=#true {
    color yolk
}
```

### `el.Shape`, `insert_shape`, `insert_asset` to `object_3d`
OLD
```python
ball_mesh = world.insert_asset(el.Mesh.sphere(0.2))
ball_color = world.insert_asset(el.Material.color(0.1, 0.2, 1.0))
ball_shape = el.Shape(ball_mesh, ball_color)
```
NEW
```kdl
object_3d ball.world_pos {
    sphere radius=0.2 {
        color 25 50 255 // red green blue [alpha]
    }
}
```

### `world.glb()` to `glb`
OLD
```python
rocket_mesh = world.glb("https://storage.googleapis.com/elodin-assets/rocket.glb"),
```
NEW
```kdl
object_3d rocket.world_pos {
    glb path="https://storage.googleapis.com/elodin-assets/rocket.glb"
}
```
## Unsupported in 0.15.0
There are a few APIs that were removed from 0.14.2 that are not yet supported.
- `el.BodyAxes`
- `el.VectorArrow`
