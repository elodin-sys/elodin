+++
title = "Why ECS for Physics?"
description = "Why ECS for Physics?"
draft = false
weight = 104
sort_by = "weight"
template = "get-started/page.html"

[extra]
toc = true
top = false
icon = ""
order = 4
+++

At Elodin's core is a physics toolkit that utilizes the ECS design pattern.
This toolkit allows our users to create incredibly efficient physics simulations easily.
Elodin's ECS engine is unique in that it does all the computation using [XLA](https://openxla.org/xla), a compiler for linear algebra.
Our engine allows your simulations to run on GPU, TPU, or CPU, all with no code changes.

## A History Lesson on ECS


ECS stands for Entity Component System, a design pattern that has recently come into vogue. It is most used in the context of video games, but as we will find out, it has been use in other places. I'm going to tell a slightly falsified version of the history of ECS to motivate its existence. The first thing to understand is how video games, simulations, and many other software pieces were historically developed using a vaguely object-oriented model. The player would be a class like:

```swift
struct Player {
   var pos: Vector3<f64>
   var vel: Vector3<f64>
}
```


Enemies would be another class in a similar vein. Maybe you would share some attributes between similar types of objects using inheritance, like:

```swift
struct GameObject {
   var pos: Vector3<f64>
   var vel: Vector3<f64>
}
struct Player: GameObject {
  var inventory: [InventoryItem]
}
struct Enemy: GameObject {}
```


This way of modeling virtual worlds can feel natural since it matches how we often think about the world. But it has some serious downsides. One key problem is that the world does not have a strict hierarchy. Oftentimes, two things may share a similar behavior but are unrelated in the traditional sense. The other issue is performance. The game object style stores all of an object's data together in a single class, often in the heap. At first, this sounds like an advantage since you can load a single object's data together at the same time. But it has some serious downsides. What if I want to integrate the velocity of all the `GameObjects` and add it to the position? I will have to write a for loop over all the GameObjects in the world and run the integrate function each time. Each time that function runs, the CPU loads the required fields from RAM and into the CPU cache. But, the CPU doesn't just load the needed fields; it loads all the memory around the `GameObject` into the cache. This process is called cache prefetching. Ideally, all the GameObjects would be loaded into the cache at once since accessing RAM is slow. But it's time-consuming for the CPU to do that since they are all spread around the heap and are usually large. The next issue is that it is difficult for systems like this to utilize vector instructions effectively. Vector instructions allow you to run the same computation many times in parallel.

ECS has taken the game development world by storm because it solves both problems simply and elegantly. ECS is best visualized as a table

| Entity ID | Pos   | Vel    | Health | Inventory     |
| --------- | ----- | ------ | ------ | ------------- |
| 1         | 0,0,0 | 10,0,0 | 1.0    |               |
| 2         | 1,1,2 | 0,0,0  | 0.0    | [Apple, Pear] |


In ECS, you have entities, components, and systems. Entities are collections of components and an ID. Components are individual properties associated with an entity, like position, velocity, or health. In an ECS system, the integrated example from above would look like this:

```rust
fn integrate(query: Query<(Pos, Vel)>) {
  for (mut pos, vel) in query {
    pos += DELTA_T * vel
  }
}
```


The code above describes a system, basically a function that operates on a set of components. When this function is run, the computer prefetches all the positions and all the velocities at once. This means that while `integrate` is running, you never have any cache-missing. It also means that you can effectively utilize vector instructions to speed up the computation. Below is a hypothetical implementation of `integrate`, where you interact with the entire array at once.

```rust
fn integrate(pos: Array<Pos> vel: Array<Vel>) {
  pos += DELTA_T * vel;
}
```

The clean version of the ECS story is that it has become a more appealing technology as CPU and GPU speed has rapidly outpaced the speed of memory. High-performance applications are usually dominated not by the actual computation time but by memory waits.

The true history is that ECS has likely been developed numerous times independently. Some sources (i.e., Wikipedia) cite the 1998 game Thief: The Dark Project as the first time it was used, but they were largely popularized roughly ten years later in a series of blog posts by Adam Martin. Recently, they are starting to gain traction outside of game development.

## An ECS-based Configurable Physics Engine


ECS isn't just great for video games; it is widely applicable to all kinds of applications. In particular, simulations are a perfect fit for ECS. Simulations benefit from both the organization and performance aspects of ECS; after all, what are video games but complex not very realistic simulations?

We want to build an ECS-based physics engine that utilizes all of ECS's benefits and is easy for non-professional software engineers to use. Historically, writing code that utilizes vectorized operations was a harrowing manual process. Engineers had to write code in an obscure, often difficult-to-read way ([https://mcyoung.xyz/2023/11/27/simd-base64/](https://mcyoung.xyz/2023/11/27/simd-base64/)). While the results were very performant, they were very readable. Simulation code is already difficult to understand due to its heavy reliance on often obscure mathematical methods. Adding complex performance optimizations to that is a recipe for confusion.

Thankfully, there is another math-heavy field that deals with this exact problem – Machine Learning (ML). There are several systems built for machine learning that allow efficient vectorized operations without sacrificing readability. We will focus on two related projects, JAX and XLA. XLA is a compiler that compiles linear algebra operations to GPU, CPU, or TPU. JAX is a JIT for Python that turns standard numpy operations into XLA intermediate representation called StableHLO – technically, it uses MHLO, which is a compatible but distinct IR to StableHLO, which is then converted by XLA into StableHLO. JAX is wonderful because it allows anyone comfortable with Numpy to write efficient code for the GPU. Python is quickly becoming the lingua franca of scientific programming, so this feature is of particular interest.

Our simulation engine works by merging ECS and Jax into a unified platform. We have ported a subset of Jax's features to Rust, a system we call Nox. Here is an example of how to write a basic six DOF (degrees-of-freedom) physics engine using Nox and our ECS.

```rust
#[derive(Clone, Component, Default)]
struct Pos(SpatialPos<f64>);
#[derive(Clone, Component, Default)]
struct Vel(SpatialMotion<f64>);
#[derive(Clone, Component, Default)]
struct Accel(SpatialMotion<f64>);
#[derive(Clone, Component, Default)]
struct ExternalForce(SpatialForce<f64>);
#[derive(Clone, Component, Default)]
struct Inertia(SpatialInertia<f64>);
#[derive(Archetype, Default)]
struct Body {
    pos: Pos,
    vel: Vel,
    accel: Accel,
    external_force: ExternalForce
}

fn calculate_accel(query: Query<(Inertia, ExternalForce)>) -> ComponentArray<Accel> {
  query.map(|inertia, force| {
    Accel(force / inertia)
  })
}

let world = World::default();
world.spawn(Body {
  pos: Pos(SpatialPos::linear(1.0, 0.0, 0.0))
  external_force: ExternalForce(SpatialForce::linear(0.0, -9.8, 0.0))
  inertia: Inertia(SpatialInertia::mass(1.0))
  ..Default::default()
});

let exec = world.build(calculate_accel.rk4::<(Pos, Vel), (Vel, Accel)>());
exec.run();
```


Let's walk through what's happening here, and how it works.
The first code block is dedicated to setting up all the components required for a six-dof system.
You'll see references to `SpatialPos`, SpatialMotion, and SpatialForce.
These are from Featherstone's Spatial Vector Algebra and are a compact way of representing the state of a rigid body with six degrees of freedom.
You can read a short into [here](https://homes.cs.washington.edu/~todorov/courses/amath533/FeatherstoneSlides.pdf) or in [Rigid Body Dynamics Algorithms (Featherstone - 2008)](https://link.springer.com/book/10.1007/978-1-4899-7560-7).

Next, you see the `Body` struct, which is an `Archetype`, which is a collection of different components that will be found together in an ECS. It allows us to initialize all the components together.

Then you can see `calculate_accel`, a function that takes the external forces and torques (SpatialForce contains both) acting on an object and converts them to accelerations.

Then we spawn a single body into the `World`, a collection of entities and associated components.

Then, we build an executable from that world and our systems. You can see that we call a method called `rk4` on `calculate_accel`. RK4 refers to Runga Kutta 4, a widely used integrator for differential equations. RK4 works with differential equations in the form:

$$
\frac{du}{dt}=f(t,u),\quad u(t_0)=u_0
$$

 In the function call, you can see two generic parameters (2 tuples). The first specifies which components make up $u$ and the second specifies which components make up $\frac{du}{dt}$

As you can see, we have built a fairly simple but complete physics simulation in 34 lines of code. It can run on GPU, CPU, or TPU (Tensor Processing Unit).

Running seamlessly on GPUs is not the only nice benefit of our ECS-based Physics engine. You can also see the type-safety that coding like this provides. Math-heavy code can easily fall victim to type errors when everything is stored in untyped matrices. After all, position and velocity are both just vectors of floats and especially when combined with dense shorthand notation, the room for error is immense. Our ECS system uses typing to guide the reader and writer, ensuring that type errors do not occur.

We plan to provide several prebuilt simulation pipelines and components, including a version of the popular MuJoCo algorithm and the algorithm used in the Basilisk Astrodynamics framework.
