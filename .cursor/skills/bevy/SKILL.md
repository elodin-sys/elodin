---
name: bevy
description: Tips for working with a Bevy application
---

# Tips for Bevy Applications

> Bevy is an Entity Component System game engine built in Rust.
> https://github.com/bevyengine/bevy | Apache-2.0 and MIT licensed

* * *


# Bevy Performance Tips

Short patterns to help the Elodin Editor stay responsive. 

(This doc was written with Elodin targeting Bevy 0.17.)



## Profiling Bevy systems with Tracy

Before trying to optimize anything, it is a good practice to measure its performance first and find the hot spots. You might turn a section of code into a zero-alloc, no clone, and it may have negligible impact because it is only called once. See the [Elodin Tracy skill document](../elodin-tracy/SKILL.md) for how to prepare and run the profiler with Elodin.

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

Reuse the same heap storage, system-private. If it can not be system-private, create a `Resource`.

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

### Beware `format!` in systems

You should also be wary of strings. Any call to `format!` allocates a string.
```rust
fn system_c(windows: Query<Entity, With<&Window>>) {
    for id in &windows {
        let s = format!("ID is {}", id);
    }
}
```
You can use `Local<String>` as I did before but for this string:

```rust
fn system_d(windows: Query<Entity, With<&Window>>,
            mut s: Local<String>) {
    for id in &windows {
        s.clear()
        let _ = write!(s, "ID is {}", id);
    }
}
```

But maybe you're not even in a system. Maybe you're in some other Rust code, then you can still minimize your allocations doing something like this:

```rust
fn deep_dark_code(...) {
    let mut s = String::new();
    for id in &windows {
        s.clear();
        let _ = write!(s, "ID is {}", id);
    }
}
```

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

## Derived query filters: Bundle up your filter

This is not a performance tip per se, but an ergonomic tip when using query filters.

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

### After with `run_system_cached`

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

### After with `run_if`

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

There is a caveat to message reading. Message reading buffers for two frames, which means if you only read every other frame, you will still get all the messages. However, if you have a system like the one below that early exits on a condition, then you may get messages you did not expect.

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

If a system A emits message M0 that is important for frame 0 and only frame 0 but `maybe_read` does not read the message, the message will persist to the next frame where it was not emitted, which can have frustrating effects. 

```
frame 0: A emits M0 -> maybe_read(false)
frame 1: A -> maybe_read(true) reads M0
```
How bad can it be? The author had a spurious input bug that persisted for over a year due to a case like this: The colon ':' key would pull up a text field, and sometimes that text field would be polluted with a ':' as its first character.

```
frame 0: A emits "enter key pressed" -> maybe_read(false) -> B shows dialog for delete file?
frame 1: A -> maybe_read(true) reads "enter key pressed" -> B deletes file
```

### Before

Consider an example where we do work when a PowerUp is added. Initially we visit every power-up on every frame even when the only reason to refresh them is an occasional input (here, a key press).

```rust
#[derive(Component)]
struct PowerUp;

fn poll_power_ups(query: Query<Entity, With<PowerUp>>, keyboard: Res<ButtonInput<KeyCode>>) {
    for entity in &query {
        if keyboard.just_pressed(KeyCode::KeyR) {
            // Check entity....
        }
    }
}
```

### After with Event

Pressing R triggers a custom [`Event`](https://docs.rs/bevy/0.17.3/bevy/ecs/event/trait.Event.html). An observer runs immediately and performs the same refresh work for all power-ups.

```rust
#[derive(Component)]
struct PowerUp;

#[derive(Event)]
struct RefreshPowerUps;

fn detect_refresh_key(mut commands: Commands, keyboard: Res<ButtonInput<KeyCode>>) {
    if keyboard.just_pressed(KeyCode::KeyR) {
        commands.trigger(RefreshPowerUps);
    }
}

fn refresh_all_on_event(_: On<RefreshPowerUps>, query: Query<Entity, With<PowerUp>>) {
    for entity in &query {
        // Check entity....
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_observer(refresh_all_on_event)
        .add_systems(Update, detect_refresh_key)
        .run();
}
```

### After with Message

Pressing R enqueues the same intent as a custom [`Message`](https://docs.rs/bevy/0.17.3/bevy/ecs/message/trait.Message.html). A normal system reads it during the schedule, so handling order matches system ordering instead of running inline at the trigger site.

```rust
#[derive(Component)]
struct PowerUp;

#[derive(Message)]
struct RefreshPowerUps;

fn detect_refresh_key(mut writer: MessageWriter<RefreshPowerUps>, keyboard: Res<ButtonInput<KeyCode>>) {
    if keyboard.just_pressed(KeyCode::KeyR) {
        writer.write(RefreshPowerUps);
    }
}

fn refresh_on_message(
    mut reader: MessageReader<RefreshPowerUps>,
    query: Query<Entity, With<PowerUp>>,
) {
    for message in reader.read() {
        for entity in &query {
            // Check entity....
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_message::<RefreshPowerUps>()
        .add_systems(Update, (detect_refresh_key, 
                              refresh_all_on_message).chain())
        .run();
}
```

In general using messages is recommended over events since their handling is better controlled and they're easier to reason about unless there is an overriding concern or ergonomics.

# Bevy Async

Use `bevy_defer` for async.

A lot of state was kept in structs to essentially handle asynchronous, multi-frame operations. Many a bespoke state machine was made. If you are bit by a case like this in the future, and don't have a suitable async solution like `bevy_defer`, I'd recommend doing something like this:

```rust
enum StateMachine {
     A,
     B { a: u32 },
     C { a: u32, b: String },
     D { a: String },
     E,
}

fn poor_mans_async(state: &mut StateMachine) {
    match state {
        StateMachine::A => {
            *state = StateMachine::B { a: 0 };
        }
        StateMachine::B { a } => {
            *a += 1;
            if a >= 100 {
                *state = StateMachine::C { a, b: String::from("hi") };
            }
        }
        // ...
    }
}

fn run_poor_mans_async() {
    let mut state = StateMachine::A;
    while state != StateMachine::E {
        poor_mans_async(&mut state);
    }
}
```

The above is essentially what async writes for us when we write code that looks like the following:

```rust
async fn rich_mans_async() -> Result<(), AccessWorld> {
     let mut a = 0;
     AsyncWorld.yield_now().await?;
     while a < 100 {
         a += 1;
         AsyncWorld.yield_now().await?;
     }
     let b = String::from("hi");
     // ...
}
```

Luckily, `bevy_defer` is an excellent library that allows us to access Bevy resources within an async context. It will not let you keep resources or references once you return from an async context via `.await`, so many times it'll hand you a Bevy resource to a closure to ensure that no references to it are kept between `.await`s.

Use `bevy_defer` if you have an operation that runs asynchronously over multiple frames that requires timing or coordination.

# Bevy Ergonomics

## Avoid type names that contain "Secondary" or "Primary".

We had two distinct code paths: one for the primary window, and another for secondary windows. This was codified in the type names that would sometimes impede them from code reuse. I have tried to unify these things where appropriate. 

## Consider using an event or message if your enum has a no-operation variant.

We had a `RelayoutWindowPhase` which is used to move windows to screens and change their dimensions. It had an `Idle` variant; `Idle` did nothing. In such cases it may be the case that you want to fire an event or send a message to have it do something.

## Prefer field access to bare accessory methods.

It was considered good practice in OO to always shield access to fields via a method or property accessor.

```rust
struct A {
    a: usize,
    b: u32,
}

impl A {
    fn a(&self) -> usize {
        self.a
    }

    fn a_mut(&mut self) -> &mut usize {
        &mut self.a
    }
}
```
If you have bare accessors like the above, it is preferred to increase the visibility of your fields to `pub(crate)` or `pub` and manipulate the fields directly. It's clearer in the code what's happening. It's more performant. Many things that OO accessors aimed to guard against can't happen in Rust:

a. No one can stick `NULL` where some other value ought to be.
b. No one can write into your value unless they have a `&mut` or owned value.

One exception to this preference is when implementing traits, which cannot express field constraints.

## Prefer .chain() to many .before() and .after() constraints in scheduling.

If you have systems that look like this:
```rust
app
    .add_systems(Update, a.before(b))
    .add_systems(Update, b.before(c))
    .add_systems(Update, c.before(d))
    .add_systems(Update, d.before(e));
```
Consider using a chain instead.
```rust
app
    .add_systems(Update, (a,
                          b,
                          c,
                          d).chain());
```
