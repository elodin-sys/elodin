# nox-ecs-macros

## Description
**nox-ecs-macros** saves you from writing verbose manual `impl`s to hook your structs into the **nox / nox-ecs** engine:  
- Declare a `Component`, assemble an `Archetype`, group components, and convert to/from computational expressions (`Noxpr`).  
- You write a struct → you add `#[derive(...)]` → it’s wired.

---

## The problem without this crate
In **nox/nox-ecs**, doing things “by hand” means:

- implementing `impeller2::component::Component` (name, schema, etc.),
- implementing `nox_ecs::Component`, `Archetype`, `ComponentGroup`,
- handling `IntoOp` / `FromOp` conversions (to/from `Noxpr`),
- writing boilerplate to insert into the `World`, initialize systems, etc.

→ So a lot of repetition and fragile code.

---

## What the Derives automate

- **`#[derive(Component)]`**  
  Turns a struct (newtype or multi-field) into a recognized ECS component
  (stable name, Impeller schema, required `nox` traits).

- **`#[derive(Archetype)]`**  
  For a struct with named fields, automatically aggregates components  
  and knows how to insert itself into a `World`.

- **`#[derive(ComponentGroup)]`**  
  Turns a struct into a logical group of components  
  (init, introspection, iteration) and can produce a `Noxpr::tuple(...)`.

- **`#[derive(IntoOp)]` / `#[derive(FromOp)]`**  
  Handles object ↔ computation conversions (to/from `Noxpr`) without boilerplate.

- **`#[derive(FromBuilder)]`**  
  Reconstructs a struct from a `nox::Builder`  
  (handy for binding system inputs).

- **`#[derive(ReprMonad)]`**  
  For 1-field wrappers parameterized by `R: Repr`,  
  provides the `nox` “representation monad” interface  
 

---

## Some examples
Runnable examples are available in the [nox-ecs README](../nox-ecs/README.md).
