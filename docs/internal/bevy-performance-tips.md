# Bevy performance tips

Short patterns that help the Elodin Editor stay responsive. 

(This doc was written with Elodin targeting Bevy 0.17.)

## Use `Local<T>` for heap-backed, per-system state

[`Local<T>`](https://docs.rs/bevy/0.17.3/bevy/ecs/system/struct.Local.html) is system-local storage: it is not a global [`Resource`](https://docs.rs/bevy/0.17.3/bevy/ecs/system/trait.Resource.html), but it persists across invocations of the same system instance. Use it for `HashMap`, `HashSet`, `Vec`, and other heap allocations that are private to one system and should not clutter the world.

### Before

Fresh allocation every frame.

```rust
fn collect_targets(mut commands: Commands, candidates: Query<Entity, With<Target>>) {
    let mut buf = Vec::new(); // This allocates on every run.
    for e in &candidates {
        buf.push(*e);
    }
    // ...
}
```

### After

Reuse the same heap storage, system-private.

```rust
fn collect_targets(
    mut commands: Commands,
    candidates: Query<Entity, With<Target>>,
    mut buf: Local<Vec<Entity>>,
) {
    buf.clear();
    buf.extend(candidates.iter());
    // ...
}
```

Note: `Vec::new()` does not allocate but the first insertion does. Thus a `Vec` on a seldom executed branch, can be left as-is.

## Query filters: Limit what is evaluated

[`QueryFilter`](https://docs.rs/bevy/0.17.3/bevy/ecs/query/trait.QueryFilter.html) types (`With`, `Without`, `Added`, `Changed`, `Or`, tuples, etc.) narrow which entities match.

### Before

No filter: every entity with `Transform` is visited every frame, even when nothing moved.

```rust
fn sync_world_labels(transforms: Query<(Entity, &Transform)>) {
    for (entity, transform) in &transforms {
        // Update label positions for every entity on every frame.
        update_labels(entity, transform.translation);
    }
}
```

### After

Only entities whose `Transform` changed this frame.

```rust
fn sync_world_labels(transforms: Query<(Entity, &Transform), Changed<Transform>>) {
    for (entity, transform) in &transforms {
        // Update label positions when transform changes.
        update_labels(entity, transform.translation);
    }
}
```

Note: When you check for `Changed<Transform>`, be aware that the display position could change due to it being in a scene hierarchy, e.g., its parent's `Transform` could have changed. If you want to ensure you capture any change of position, no matter where it comes from, use `Changed<GlobalTransform>` and check the `GlobalTransform` which will have the display position of the object.

## Derived query filters: Package up your filter

This is not a performance tip per se, but more of an ergonomic tip when using query filters.

### Before

Long filter tuples repeated at every `Query` site.

```rust
fn system_a(q: Query<Entity, (With<Alive>, With<Player>)>) { /* ... */ }
fn system_b(q: Query<&Name, (With<Alive>, With<Player>)>) { /* ... */ }
```

### After

One derived [`QueryFilter`](https://docs.rs/bevy/0.17.3/bevy/ecs/query/trait.QueryFilter.html).

```rust
#[derive(QueryFilter)]
struct ActivePlayer {
    alive: With<Alive>,
    player: With<Player>,
}

fn system_a(q: Query<Entity, ActivePlayer>) { /* ... */ }
fn system_b(q: Query<&Name, ActivePlayer>) { /* ... */ }
```

## One-off work: `Commands::run_system_cached` or `run_if`

Use [`Commands::run_system_cached`](https://docs.rs/bevy/0.17.3/bevy/ecs/system/struct.Commands.html#method.run_system_cached) (or [`World::run_system_cached`](https://docs.rs/bevy/0.17.3/bevy/ecs/world/struct.World.html#method.run_system_cached)) when heavy work should run only on demand (save, import, palette action), not every frame. Bevy reuses cached system state for the same system type, so repeated invocations avoid paying full setup each time.

The common mistake is the name `run_cached_system`—the API is `run_system_cached`.

### Before

A system on `Update` that runs every frame; most of the time it immediately returns, but you still pay scheduling and system-param fetch for work that is only needed occasionally.

```rust
use bevy::prelude::*;

// This example omits other SystemParams such as queries and resources.
fn save_if_requested(keyboard: Res<ButtonInput<KeyCode>>,
                     // SystemParams required to save.
                     query: Query<&Saveables>,
                     file: Res<SaveFile>,
                     // ...
                     ) {
    if !keyboard.just_pressed(KeyCode::KeyS) {
        return;
    }
    // Rebuild buffers, write files, and so on.
    // This path should run rarely, but the system still runs every frame.
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Update, save_if_requested)
        .run();
}
```

### After

A cheap per-frame system only checks input; heavy systems run only when needed, via `run_system_cached`.

```rust
use bevy::prelude::*;

fn detect_save_shortcut(keyboard: Res<ButtonInput<KeyCode>>, mut commands: Commands) {
    if keyboard.just_pressed(KeyCode::KeyS) {
        commands.run_system_cached(save_to_disk);
    }
}

fn save_to_disk(query: Query<&Saveables>,
                file: Res<SaveFile>,
                // ...
                ) {
    // Heavy work: Flush serialized tiles to disk here.
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Update, detect_save_shortcut)
        .run();
}
```

### Better Still `run_if`

An even better means of achieving the above is to use the `run_if`, which when it returns false the `SystemParam`s for `save_to_disk` are not evaluated. 

```rust
use bevy::input::common_conditions::input_just_pressed;
use bevy::prelude::*;

fn save_to_disk(
    query: Query<&Saveables>,
    file: Res<SaveFile>,
    // ...
) {
    // Heavy work: flush serialized tiles to disk here.
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Update,
            save_to_disk.run_if(input_just_pressed(KeyCode::KeyS)),
        )
        .run();
}
```
Note: `run_if` accepts any system that returns a boolean.

```rust
fn save_pressed(keys: Res<ButtonInput<KeyCode>>) -> bool {
    keys.just_pressed(KeyCode::KeyS)
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Update, save_to_disk.run_if(save_pressed))
        .run();
}
```

## Slower-than-display-rate work: custom schedules vs frame pacing

Not everything needs to run every frame at display refresh (often ~60 Hz).

### Before

Heavy work on every `Update` tick.

```rust
use bevy::prelude::*;

fn expensive_remote_poll() {
    // Network, database, or aggregation work runs more than sixty times per second.
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Update, expensive_remote_poll)
        .run();
}
```

### After

Same system, throttled with [`on_timer`](https://docs.rs/bevy_time/0.17.3/bevy_time/common_conditions/fn.on_timer.html).

```rust
use bevy::prelude::*;
use bevy_time::common_conditions::on_timer;
use std::time::Duration;

fn expensive_remote_poll() {
    // This runs at most once per second.
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Update,
            expensive_remote_poll.run_if(on_timer(Duration::from_secs(1))),
        )
        .run();
}
```

Other options (no full code here):

- Dedicated schedule + throttling: Define a [`ScheduleLabel`](https://docs.rs/bevy/0.17.3/bevy/ecs/schedule/trait.ScheduleLabel.html), register systems on that schedule, and drive it from a lightweight system using [`Time`](https://docs.rs/bevy/0.17.3/bevy/time/struct.Time.html) / [`Timer`](https://docs.rs/bevy/0.17.3/bevy/time/struct.Timer.html) with [`run_if`](https://docs.rs/bevy/0.17.3/bevy/ecs/schedule/common_conditions/index.html) or by calling [`World::run_schedule`](https://docs.rs/bevy/0.17.3/bevy/ecs/world/struct.World.html#method.run_schedule) when your guard says it is time.

## Profiling Bevy systems with Tracy

[Tracy](https://github.com/wolfpld/tracy) shows where frame time goes, including Bevy schedules and individual systems, when the editor is built with tracing wired to Tracy.

In this repo:

1. Build and run the Tracy workflow from the dev shell (see [elodin-tracy skill](../../.cursor/skills/elodin-tracy/SKILL.md)): `nix develop`, `just install tracy`, then run the editor (for example `elodin editor examples/sensor-camera/main.py`).
2. Start the Tracy GUI (`tracy`) and connect to the editor client. The editor uses Tracy port 8087 by default (see the skill’s port table for render server, simulation, and DB).
3. With the `tracy` feature, `elodin-editor` enables Bevy’s `trace_tracy` integration (`libs/elodin-editor/Cargo.toml`). That instruments systems and schedules into Tracy zones without extra Bevy-specific code in your systems.
4. Rust [`tracing`](https://docs.rs/tracing) spans also show up as zones (the `elodin` binary installs a `TracyLayer`). Use `#[tracing::instrument]` or `tracing::info_span!` on hot editor code; raise `RUST_LOG` (e.g. `debug`) if spans are filtered out by the default env filter.

### Before

Hot path invisible in Tracy as its own zone.

```rust
fn rebuild_graph_cache(graph: Res<GraphLayout>) {
    // This work is expensive and appears only as parent frame time unless Bevy adds a label.
    let _ = graph.0.len();
}
```

### After

Explicit zone plus optional function name in traces.

```rust
#[tracing::instrument(skip(graph))]
fn rebuild_graph_cache(graph: Res<GraphLayout>) {
    let _ = graph.0.len();
}
```

Note: `skip(graph)` tells [`#[tracing::instrument]`](https://docs.rs/tracing/0.1/tracing/attr.instrument.html) not to attach the `graph` argument as a span field. By default the macro would try to record every parameter (usually via `Debug`), which is noisy for large values, can fail if a type has no useful `Debug`, and is rarely needed when you only want a named zone in Tracy.

Platform note: Tracy profiling for Elodin is Linux-oriented in practice (GUI, sampling, and subprocess ports). On macOS, follow the skill’s guidance (e.g. Linux host or OrbStack NixOS VM).

For capture CLI, simultaneous editor + sim traces, troubleshooting (vsync, `TRACY_NO_EXIT`, sudo for sampling), and IREE/sim tracing, use the full skill rather than duplicating it here.

## More ECS and tooling tips

### Parallel queries: only when work per entity is large enough
[`Query::par_iter`](https://docs.rs/bevy/0.17.3/bevy/ecs/system/struct.Query.html#method.par_iter) (and related parallel iterators) can use multiple threads, but scheduling and splitting have a fixed cost. Prefer them when each matching entity does enough CPU work to amortize that overhead; for tiny per-entity updates, a serial `iter()` is often faster. Profile with Tracy or frame diagnostics before leaning on parallelism.

### Before

Parallel overhead dominates cheap work.

```rust
#[derive(Component)]
struct Tint(f32);

fn apply_tints(query: Query<&mut Tint>) {
    query.par_iter_mut().for_each(|mut t| {
        t.0 *= 1.001; // There is too little work per entity to amortize parallelism.
    });
}
```

### After

Serial is often faster for tiny updates.

```rust
#[derive(Component)]
struct Tint(f32);

fn apply_tints(query: Query<&mut Tint>) {
    for mut t in &mut query {
    // OR query.iter_mut().for_each(|mut t| {
        t.0 *= 1.001;
    }
}
```

Use `par_iter` when inner work is large (physics, mesh rebuild chunks, etc.), not for a few float ops.

Of course depending on one's workload, this tip might actually go the other way: from `iter()` to `par_iter()`.

### Events vs Messages

Bevy has `Event`s and `Message`s. They both decouple "what happened" into an event or message, and "what response" should result via an observer or polling respectively. But the performance and ergonomics have some subtle distinctions. This [table](https://taintedcoders.com/bevy/events#choosing-messages-vs-events) highlights their differences.

|                         | Events                      | Messages                          |
|-------------------------|-----------------------------|-----------------------------------|
| Optimal event frequency | Infrequent                  | Frequent                          |
| Handler                 | Only handles a single event | Can handle many messages together |
| Latency                 | Immediate                   | Up to 1 frame                     |
| Event propagation       | Bubbling                    | None                              |
| Scope                   | World _or_ Entity           | World                             |
| Ordering                | No explicit order           | Ordered                           |
| Coupling                | High                        | Low                               |

Components have [life-cycle
events](https://docs.rs/bevy/0.17.3/bevy/ecs/lifecycle/index.html): Add, Insert,
Replace, Remove, Despawn. Components also have
[hooks](https://docs.rs/bevy/0.17.3/bevy/prelude/trait.Component.html#adding-components-hooks):
`on_add`, `on_insert`, `on_replace`, and `on_remove`. The component hooks are a
tighter binding than the `Event` observer. 

### Handler Runs When for Events?
The table above can be a guide for performance considerations. One non-obvious complication of observer `Event` handling is that because it runs "immediately", its handler runs between potentially many different system boundaries. The handlers run after every system that calls `commands.trigger(event)`. With `Message`s only systems that poll `EventReader<M>` handle it, and they handle it in a consistent system order. 

Let me give one example of an app that has two systems: A, B, event E, and two observers of E: X, Y. A is called before B. So the system call graph looks like this in general (assuming single threaded):

```
A -> B
```

But in cases where A `commands.triggers(E)` then the call graph looks like this:
```
A triggers E -> X -> Y -> B
```
Note: `commands.trigger(E)` like `commands.spawn(...)` does not run immediately; it batches its operations.

Or it could look like this because observers are not ordered.
```
A triggers E -> Y -> X -> B
```
So any triggers of the event E will effectively add its handlers in some non-explicit order to the system call graph.

```
A -> B -> triggers E -> X -> Y
OR
A -> B -> triggers E -> Y -> X
```

If instead of calling `commands.trigger(E)` one calls `world.trigger(E)` then the handlers run immediately in a non-explicit order. 

### Handler Runs When for messages?

Let me give one example of an app that has two systems: A, B, message M, and a system that polls for M called X. A is called before B. So the system call graph will be one of these in general (assuming single threaded):

```
A -> B -> X
A -> X -> B
X -> A -> B
```

Let's focus on the second case `A -> X -> B` since it will illustrate the handling between frames better and say that A and B emit a message M.

```
frame 0: A emits M1 -> X handles M1 -> B emits M2
frame 1: A emits M3 -> X handles M2, M3 -> B emits M4
```

It is easier to reason about where messages are handled than where events are handled because its apparent in the system ordering for messages, while the event handling has a more ephermal quality because it can happen after any system that triggers the event.

#### Message Reading Gotcha

There is a caveat to message reading. Message reading buffers for two frames, which means if you only read every other frame, you will still get all the messages. However, if you have a system like the one below that early exits on a condition, then you may get messages you don't expect.

```rust
fn maybe_read(run: In(bool), messages: MessageReader<M>) {
  if ! run.0 {
    return;
  }
  for message in messages.read() {
    // Process message.
  }
}
```

If a system A emits message M0 and let us say it is important for frame 0 and only frame 0 but `maybe_read` does not read the message, the message will persist to the next frame where it was not emitted, which can have frustrating effects. 

```
frame 0: A emits M0 -> maybe_read(false)
frame 1: A -> maybe_read(true) reads M0
```
How bad can it be? (The author had a spurious input bug that persisted for over a year due to a case like this.)

```
frame 0: A emits "enter key pressed" -> maybe_read(false) -> B shows dialog for delete file?
frame 1: A -> maybe_read(true) reads "enter key pressed" -> B deletes file
```

### Before

Visit every matching entity every frame even when you only need one-time setup.

```rust
#[derive(Component)]
struct PowerUp;

fn poll_power_ups(query: Query<Entity, With<PowerUp>>) {
    for entity in &query {
        check_state(entity); // For example, re-check state for every power-up on each tick.
    }
}
```

### After

Handler runs when `PowerUp` is added.

```rust
use bevy::prelude::*;

#[derive(Component)]
struct PowerUp;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_observer(|event: On<Add, PowerUp>| {
            // This runs only when the component is added.
            check_state(event.target());
        })
        .run();
}
```


### Diagnostics vs Tracy

Bevy’s built-in [diagnostics](https://docs.rs/bevy/0.17.3/bevy/diagnostic/index.html) (e.g. frame time, FPS plugins where enabled) are good for quick checks in-app. Use Tracy (section above) when you need per-system breakdowns, cross-thread timelines, and correlation with Rust `tracing` spans. Pick the tool to match how deep you need to go.

### Before

No built-in frame stats.

```rust
App::new()
    .add_plugins(DefaultPlugins)
    .run();
```

### After

Frame time and FPS diagnostics for quick inspection.

```rust
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};

App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(FrameTimeDiagnosticsPlugin::default())
    .add_plugins(LogDiagnosticsPlugin::default())
    .run();
```

### Trim `bevy` features for minimal binaries

For headless tools, tests, or crates that only need a slice of the engine, depend on `bevy` with `default-features = false` and enable only the subsystems you need (see Bevy’s crate features in [`Cargo.toml`](https://docs.rs/crate/bevy/0.17.3/source/Cargo.toml)). That cuts compile time and binary size, which speeds iteration even when runtime FPS is unchanged.

### Before

```toml
bevy = "0.17"
```

### After

Illustrative—pick features to match your crate.

```toml
bevy = { version = "0.17", default-features = false, features = [
    "bevy_asset",
    "bevy_log",
    "bevy_window",
    "bevy_winit",
] }
```
