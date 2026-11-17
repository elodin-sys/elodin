# Elodin Feature Request: One-World Many-Agent Real-Time Scaling

## Request

Extend Elodin so the `many-agent-lockstep` example can run 24 controlled agents
and 24 observed objects at 1 kHz in one shared world and one packaged process
tree.

The same example must remain the performance contract. The returned
implementation should upgrade it to use the new public APIs, preserve its
canonical outputs, and include before/after receipts from the acceptance VM.

The shared-world and packaged-process-tree constraints are acceptance
requirements. Entity counts, recording policy, and the production execution
path must remain unchanged.

## Why this example exists

Elodin already stores like-typed entity state in columns and compiles mapped
systems efficiently. The remaining loop still crosses several boundaries whose
cost grows with entity and component count:

1. The runtime scans DB-backed pairs before graph execution.
2. The graph executes and produces current world values.
3. The runtime scans and commits world values.
4. `post_step` reconstructs current state through the DB by string name.
5. Python prepares frames and applies responses serially.
6. Callback writes return through the DB and enter the world on the next cycle.

The benchmark separates graph, DB, callback, and external exchange costs so an
optimization cannot move work into an unmeasured phase.

## Measured current baseline

The example was measured on a 22-logical-CPU x86_64 Linux development machine.
These values are forecast evidence, not certification.

At 10+10, paced lockstep produced:

- 734.8 effective cycles/s;
- 1.361 ms mean cycle;
- 878.1 µs graph mean and 1.656 ms estimated p95;
- 31.8 µs DB pull mean;
- 67.7 µs commit mean;
- 381.8 µs callback mean and 896.2 µs estimated p95.

At 24+24, paced lockstep produced:

- 330.0 effective cycles/s;
- 3.030 ms mean cycle;
- 1.845 ms graph mean and 3.620 ms estimated p95;
- 127.0 µs DB pull mean;
- 287.9 µs commit mean;
- 767.9 µs callback mean and 1.195 ms estimated p95;
- every measured cycle late and 3,008 pacing resets.

The three-repeat unpaced matrix confirmed the same scaling shape. At 10+10,
median cycle time was 1.650 ms with 1.027 ms graph, 49.7 µs pull, 90.0 µs
commit, and 480.1 µs callback. At 24+24, median cycle time was 3.304 ms with
1.960 ms graph, 155.3 µs pull, 336.7 µs commit, and 848.8 µs callback. Exact
output parity passed across repeats, local/lockstep modes, and count prefixes.

The 24+24 case therefore needs approximately:

- 3.02× higher effective throughput;
- 2.75× lower mean cycle;
- 4.61× lower graph mean and 9.05× lower graph p95;
- 1.28× lower callback mean and 1.20× lower callback p95;
- a combined DB and runtime-overhead reduction sufficient to fit graph,
  callback, pull, commit, and framework work inside 1.1 ms.

The unpaced matrix, receipts, output digests, and phase-slope forecast are
generated under `profile_output/many-agent-lockstep-baseline/`. Paced 10+10 and
24+24 receipts are under `profile_output/many-agent-lockstep-paced/`. Traced
counterparts are under `profile_output/many-agent-lockstep-profile/`.

## Current public API limitations

### StepContext resolves and copies every cycle

`StepContext.component_batch_operation` improves lock acquisition, but each
call still:

- accepts strings and hashes component names;
- resolves DB pairs rather than prepared world columns;
- copies read bytes into Rust vectors;
- allocates Python dictionaries and NumPy arrays;
- copies write arrays back into new Rust vectors;
- appends writes to DB time series.

The operation is useful for historical reads and asynchronous clients. It is
not a zero-copy real-time world interface.

### Callback ordering forces a DB round trip

The served loop is currently:

```text
pre_step
DB pull
graph execution
world commit
write barrier
post_step
pacing
```

`post_step` therefore reads values that were just committed and writes commands
that must be pulled back into the world on the next cycle. The ordering is
correct, but the DB is used as an avoidable in-process transport.

### Synchronization remains scan based

`DbSync` caches pair resolution and payload shadows, but pull and commit still
visit every retained row. `World.dirty_components` exists, yet execution and
commit do not consume a cycle-scoped set of semantic writes.

### Recording policy is not expressive enough

Current metadata distinguishes transient values, sparse on-change recording,
and `record_every_tick`. Applications also need explicit lower-rate and manual
recording without removing schema visibility.

### Profiling is not yet a certification interface

The runtime prints useful phase summaries and can emit JSON, while deeper
Cranelift detail is coupled to debug configuration. The public API lacks stable
system identities, exact boundary percentiles, synchronization counters, and a
single versioned receipt suitable for automated acceptance.

## Feature 1: Prepared column handles

### Proposed API

```python
agent_state = world.prepare_component(
    "position",
    entities=agent_entities,
    access="read",
)
commands = world.prepare_component(
    "command",
    entities=agent_entities,
    access="write",
)
observed_state = world.prepare_component(
    "object_position",
    entities=object_entities,
    access="read",
)

exchange = world.prepare_batch(
    {
        "agent_state": agent_state,
        "observed_state": observed_state,
        "commands": commands,
    },
    join="strict",
)
```

Suggested types:

```python
class ComponentHandle:
    component_id: ComponentId
    access: Literal["read", "write"]
    dtype: numpy.dtype
    shape: tuple[int, ...]
    rows: tuple[EntityId, ...]
    topology_generation: int

class BatchHandle:
    fields: Mapping[str, ComponentHandle]
```

### Required semantics

- Resolve component IDs, dense slots, row order, dtype, and shape once.
- Store layout identities and row mappings, not raw pointers. Backend output
  buffers may be swapped.
- Reject missing rows, duplicate rows, incompatible shapes, and write access to
  non-writable columns during preparation.
- Invalidate handles when topology generation changes.
- Preserve caller-specified entity order.
- Keep views callback-scoped unless an explicit owned copy is requested.

### Expected effect

Remove repeated name hashing, DB pair lookup, Python dictionary construction,
and per-cycle shape validation from the fast path.

### Acceptance evidence

The receipt must report prepared-handle count, handle validation failures,
bytes viewed, bytes copied, and prepared versus legacy operation counts.

## Feature 2: Pre-commit world callback

### Proposed API

```python
def exchange_step(tick: int, ctx: el.WorldStepContext) -> None:
    with ctx.batch(exchange) as view:
        view.commands[...] = controller.step_batch(
            view.agent_state,
            view.observed_state,
        )

world.run(
    system,
    pre_commit=exchange_step,
)
```

Suggested context:

```python
class WorldStepContext:
    tick: int
    timestamp: int

    def view(self, handle: ComponentHandle) -> numpy.ndarray: ...
    def batch(self, handle: BatchHandle) -> BatchView: ...
    def mark_dirty(self, handle: ComponentHandle, rows=None) -> None: ...
    def record(self, handle: ComponentHandle, rows=None) -> None: ...
```

### Required ordering

```text
DB pull
graph execution
pre_commit
world commit
post_step
pacing
```

The callback sees current graph outputs. Its writes are included in the same
commit and become graph inputs on the next cycle. This preserves one-cycle
command causality without using DB writes as an internal transport.

### Correctness constraints

- Mutable view acquisition conservatively marks the complete column dirty.
  Optional row masks may narrow this later.
- Conflicting graph and callback writers are rejected during world build.
- Hook exceptions abort the run by default and produce a failed receipt.
- Historical/time-indexed reads remain on the DB-backed `StepContext`.
- The callback frequency is explicitly defined in telemetry-commit units.
  Per-internal-tick callbacks, if needed, must be a separate API.

### Expected effect

Remove one DB read path and one DB write path from every control cycle while
retaining recording and next-cycle behavior.

## Feature 3: Native batched hook and controller exchange

Python should not be required to loop over every agent when the work is already
fixed-shape and native.

### Proposed API

```python
hook = el.NativeStepHook.load(
    "libmany_agent_hook.so",
    input=exchange,
    symbol="many_agent_step",
    workers=8,
)

world.run(system, pre_commit=hook)
```

Equivalent Rust interface:

```rust
pub trait PreCommitHook: Send {
    fn prepare(&mut self, layout: &PreparedLayout) -> Result<(), HookError>;
    fn step(&mut self, step: &mut WorldStep<'_>) -> Result<ChangeSet, HookError>;
}
```

### Required capabilities

- Pass prepared contiguous column views without Python object construction.
- Allow one native invocation to submit all controller work, wait concurrently,
  and merge responses in stable entity order.
- Support a fixed native worker pool and caller-provided process/socket
  integration.
- Release the Python GIL for native hooks.
- Include hook wall time, queue time, worker occupancy, and response tail in
  the standard receipt.
- Keep a pure Python hook available for development and parity testing.

### Expected effect

Eliminate per-agent Python frame preparation, response decoding, and callback
dispatch while retaining one world and isolated controller processes.

## Feature 4: Explicit ownership and cycle change sets

### Proposed API

```python
el.Component(
    "command",
    el.ComponentType(el.PrimitiveType.F64, (3,)),
    owner=el.Owner.pre_commit(),
    recording=el.Recording.on_change(),
)
```

Owners:

- `system`: written by compiled graph systems;
- `external`: written by DB/Impeller clients;
- `pre_commit`: written by the direct hook;
- `static`: initialized once and read-only.

Existing `external_control=true` metadata should map to `external`.

### Execution metadata

```rust
pub struct ExecMetadata {
    pub input_slots: Vec<ExecSlotMetadata>,
    pub output_slots: Vec<ExecSlotMetadata>,
    pub semantic_reads: Vec<ComponentId>,
    pub semantic_writes: Vec<ComponentId>,
    pub systems: Vec<SystemExecMetadata>,
}
```

Do not infer semantic writes from merged ABI return slots alone. Merged systems
may return pass-through values. Preserve declared subsystem writes before
pipeline merging.

### Cycle result

Execution should return a `ChangeSet` containing:

- semantic graph outputs;
- mutable hook views;
- explicit row writes;
- recording policies due this commit.

Commit should consider only this union, then retain byte comparison for sparse
on-change recording.

### Compatibility

- Keep legacy shared ownership as the default during migration.
- Offer an opt-in planned synchronization mode.
- Fail world construction on ambiguous multiple writers in planned mode.
- Preserve truncation and asynchronous external-write behavior.

## Feature 5: Change-driven DB synchronization

### Proposed DB API

```rust
let cursor = db.watch_components(ids)?;

cursor.changed().await;
for changed in cursor.drain_latest() {
    // Apply the newest complete value.
}
```

### Runtime behavior

- Pull only externally owned pairs changed since the cursor advanced.
- Coalesce repeated writes to the newest value.
- Commit only candidates in the cycle `ChangeSet`.
- Replace write-barrier polling with cursor notification.
- Force complete resynchronization after truncation or topology change.

### Required counters

- rows registered;
- rows scanned;
- changed rows drained;
- candidate rows compared;
- rows committed;
- unchanged writes suppressed;
- bytes copied in each direction;
- cursor wakeups and coalesced writes.

The benchmark expects zero complete pull scans when external inputs are
unchanged and commit candidates proportional to changed or due columns.

## Feature 6: Typed recording policies

### Proposed API

```python
el.Recording.on_change()
el.Recording.every_commit()
el.Recording.every(10)
el.Recording.manual()
el.Recording.initial_only()
el.Recording.transient()
```

### Semantics

- Cadence is expressed in telemetry commits.
- `on_change` preserves byte-exact sparse recording.
- `every_commit` maps from `record_every_tick=true`.
- `every(n)` records the current value every `n` commits.
- `manual` records only when requested by the callback.
- `initial_only` registers schema and one initial value.
- `transient` remains outside DB schema, history, replication, and editor
  discovery.

Conflicting typed and legacy settings should fail world construction.

### Expected effect

Retain useful live and replay evidence without forcing every diagnostic column
through every commit.

## Feature 7: Structured timing and stable system identity

### Proposed API

```python
world.run(
    system,
    timing=el.Timing.summary(),
    on_receipt=consume_receipt,
)

receipt = execution.last_receipt()
```

Timing levels:

- `summary`: complete-cycle phases with minimal overhead;
- `systems`: stable named-system inclusive/exclusive times;
- `operations`: compiler/backend detail with measured perturbation.

Stable system naming:

```python
@el.system(name="agent_dynamics")
def agent_dynamics(...):
    ...
```

Without an explicit name, use source location and `__qualname__`, not
`repr()`.

### Receipt contents

- schema version and source revision;
- requested/executed ticks and warmup exclusion;
- count, sum, min, exact p50/p95/p99, and max per phase;
- graph, DB pull, pre-commit, commit, post-step, pacing, and total cycle;
- synchronization counters and byte counts;
- backend input/output bytes and kernel invocation count;
- compile/lowering/code-generation phases;
- named-system times when requested;
- timing collection overhead and unavailable fields.

Instrumentation and checkpoint/debug artifact generation must be independently
selectable.

## Benchmark and acceptance

The current example must be measured before and after the API work.

Characterization matrix:

```text
1, 2, 4, 8, 10, 12, 16, 20, 24 agents
equal observed-object count
2,000 warmup ticks
10,000 measured ticks
3 balanced-order repeats
```

Acceptance case:

```text
24 agents + 24 observed objects
1 kHz graph and commit cadence
60,000 measured ticks
5 repeats
one shared world
one packaged process tree
```

Every acceptance repeat on an exclusive x86_64 Linux VM must satisfy:

- effective rate at least 995 Hz;
- mean cycle no more than 1.1 ms;
- graph mean and exact p95 no more than 400 µs;
- callback mean no more than 600 µs;
- callback exact p95 no more than 1 ms;
- late-cycle fraction no more than 0.1%;
- zero pacing resets;
- complete deterministic and backend parity evidence;
- no missing or malformed receipt field.

AWS `c7i.8xlarge` is a public reference instance class. A documented equivalent
x86_64 Linux VM is acceptable. Run one benchmark job at a time.

Current-machine results are forecasts only. Fit graph, DB, and callback slopes
independently and report how much each phase must improve. Do not scale results
by CPU count or use projections as certification.

## Required return package

Return one relative-path archive containing:

- upgraded example source;
- source revision and clean/dirty state;
- acceptance configuration;
- before/after receipts;
- raw summary JSON and exact phase traces;
- output digests and backend parity evidence;
- sanitized VM facts;
- manifest of file sizes and SHA-256 hashes.

Exclude hostnames, usernames, absolute paths, network addresses, unrestricted
environment dumps, DB contents, and unrelated source.

The implementation is accepted only when the upgraded example itself passes.
Standalone microbenchmarks may explain improvements but cannot replace the
one-world end-to-end result.
