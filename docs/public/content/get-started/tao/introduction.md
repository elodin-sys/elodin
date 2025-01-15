+++
title = "What is Elodin"
description = "What is Elodin"
draft = false
weight = 163
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 63
+++

## Hardware is Hard
If you've ever built anything with hardware involved you know it is incredibly time-consuming. Even the simplest of projects can spiral out of control.
Never is this more true than for aerospace projects. Satellites have historically taken around 10 years to produce, and even now it can take over a year to produce a simple CubeSat. Drones are usually much quicker to produce, but can still take over a year to make a fully featured drone platform. Rockets, airplanes, and helicopters take this to an even bigger extreme.

There are some excellent reasons these projects take so long to produce; they are complex – typically made by large teams of people, often breaking new ground on technology. That's the inherent complexity, the complexity that makes these problems worth solving. There is also the incidental complexity, the complexity that is not required and just comes along for the ride. Elodin's goal is to solve that incidental complexity, starting with a simulation and test platform.


### The Elodin Way

Elodin's simulation system has 4 primary components:
- An ECS based physics toolkit that allows you to use JAX or Rust to write GPU-backed simulations.
- A cloud Monte Carlo runner, that enables users to run massive numbers of their simulations in parallel.
- A 3D viewer that allows you to visualize your simulation in realtime.
- A communication protocol that links together the simulation, flight control software, and the data-collection system.
