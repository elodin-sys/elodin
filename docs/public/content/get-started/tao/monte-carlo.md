+++
title = "Monte Carlo"
description = "Monte Carlo"
draft = false
weight = 106
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 6
+++

With our physics toolkit, you can simulate a wide variety of physical situations. But what's the point of making these simulations?
Our primary goal is to test control systems for aerospace vehicles.
It's not sufficient to test the control systems in just one situation; you need to test them across the entire gamut of possible outcomes.
To do that, you can use Monte Carlo testing. Monte Carlo is a tongue-in-cheek reference to the famous Monaco casino.
The name originated in the depths of the Manhattan Project during WW2, where they needed a clandestine name to mask the details of the technique.

As the name would suggest, Monte Carlo methods work by randomly sampling a system. In this case, the random sampling is of the control system and simulations. You randomize a set of input values to the simulation and then record outputs for each iteration.

## Monte Carlo Challenges

Monte Carlo systems are commonly used, but they can be particularly challenging for traditional simulation systems.
Most systems allow you to randomize inputs into the system, but they make it far more challenging to run many simulations in parallel.
MatLab's license model makes it prohibitively expensive to run parallel simulations.
Other solutions simply require large amounts of engineering effort and/or massive amounts of computing power to run simulations in parallel.

## Our Solution

Elodin solves these problems by introducing a cloud-based simulation runner that can run up to 100,000 simulations simultaneously. We're able to do so in a cost-effective way by:

- Using the [ECS](https://en.wikipedia.org/wiki/Entity_component_system) architecture to exploit the inherent parallelism in running the same compute kernel against different sets of data.
- Leveraging [XLA](https://openxla.org/xla) to compile the simulation code to run on GPUs.
- Using the cloud's elasticity to scale up and down as needed.
