+++
title = "Overview"
description = "Overview"
draft = false
weight = 101
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 1
+++


## Simulation Architecture

An Elodin simulation consists of Entities are spawned with a list of archetypes rather than individual components.

### Systems
Systems are reusable functions that operate on a set of components. Composing different systems allows for the creation of custom physics engines. Some examples of systems include gravity, aerodynamics, collision detection.

### World
World is a container of all the entities in the simulation. See [ECS Data Model](/reference/overview#ecs-data-model) for more context on entities.

### Flight Software (FSW)
Flight software is a set of processes that run independent of the simulation. Using `impeller`, FSW subscribes to relevant simulation data and sends control commands back to the simulation.

{% image(href="/assets/architecture") %}Simulation Architecture{% end %}


## ECS Data Model

Elodin uses the Entity Component System (ECS) pattern to manage simulation data. Entities are simply unique references to a collection of components. [Components] are individual properties associated with an entity like position, velocity, mass, etc.

[Archetypes] are a unique combination of components that allow for efficient memory management and data access patterns. Entities are spawned with a list of archetypes rather than individual components.

{% image(href="/assets/data_model") %}Data Model{% end %}

Systems are simply functions that operate on component data.

{% image(href="/assets/systems") %}Systems{% end %}

[Systems]: /reference/python-api#systems
[Components]: /reference/python-api#components
[Archetypes]: /reference/python-api#archetypes
