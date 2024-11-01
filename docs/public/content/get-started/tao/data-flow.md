+++
title = "ECS Data Flow"
description = "ECS Data Flow"
draft = false
weight = 105
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 5
+++

One of the pernicious difficulties we encountered developing the physics toolkit was how different components should communicate.
The trouble is that the physics toolkit is entirely customizable, so it's possible to have infinite possible data combinations.
Thankfully, ECS provides an exceedingly simple way of syncing data between systems.

## Impel
 We call this protocol Impel. You can send data as a tuple of the entity id, component id, and value, like so:

```rust
(EntityId(1), ComponentId("pos"), Vector3(1,1,3))
```

This might not seem impressive at first blush, but it offers some really nice benefits. Say the control software wants to update the simulated motor rpm; all it would have to do is send the following msg to the simulation software.

```rust
(EntityId(2), ComponentId("motor_rpm"), 2.0)
```

The simulation software can also send out sensor output this same way.

```rust
(EntityId(3), ComponentId("accelerometer_reading"), Vector3(0.0, -8.0, 0.0))
```

This is nice because it means reacting to new inputs and producing outputs that work with the same systems that we use for the core simulation above.

We use Impel to record the entire state of the system. We create a recorder service that subscribes to all components and entities from the simulation and control software. Every time it receives a message, it adds it to a table. Then, you can use that data to query the entire state of the system at any time. This allows users to specify failure and success conditions with ease; simply write a statement like

```rust
assert(entity[3].pos == vector3(1.0, 0.0, 0.0))
```

We can also use that data to provide real-time visualization of the simulation, graphs, and other analytics about the output. And since this is just a generic data exchange protocol, it also works for generic telemetry out of your software. Store and analyze anything you want: memory usage, performance stats, internal metrics about your Kalman filter; it doesn't matter.
