+++
title = "Custom C++ Runtimes"
description = "Export an Elodin simulation as a native library, step it from your own C++ program without Python or Elodin, and prove the two runs match"
draft = false
weight = 106
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 6
lead = "Compile the simulation you built in Elodin into a native library, run it from your own C++ program, and prove it reproduces the Elodin run byte for byte."
+++

You built and tested a simulation in Elodin, and now it has to run somewhere Elodin does not: a test rig that only runs frozen, reviewed C++; a customer's simulation framework; a contract that says the deliverable is a library and a header with no Python. This page takes the simulation from Elodin into your own C++ program, and shows how to prove the program still computes exactly what Elodin computed.

Two examples carry the walkthrough. [ball-cpp](https://github.com/elodin-sys/elodin/tree/main/examples/ball-cpp) is the smallest case: one body, a C++ loop, and a CSV that matches Elodin bit for bit. [many-agents-cpp](https://github.com/elodin-sys/elodin/tree/main/examples/many-agents-cpp) is a full SITL: 24 agents, C++ flight software, worker threads, and a witness that Elodin's own tools compare byte for byte.

You may not need this. To put flight software in the loop with an Elodin simulation, use [SITL & HITL](/reference/sitl-hitl), which keeps the editor, database and recipes. To step a simulation from Python for batch runs or reinforcement learning, [`World.to_jax`](/reference/python-api/#class-elodin-jaxsim-jaxsim) is enough.

## The Elodin way

- **One model, one compiled step.** The library holds the same Cranelift code Elodin runs in `World.run`, compiled from the same Python. You do not port the physics by hand.
- **A narrow C boundary.** Your program sees components as arrays of bytes, by name, and three verbs: read, write, step. Nothing else crosses it.
- **Your loop, your rules.** Scheduling, I/O, threads and logging belong to your program. The library does no I/O and starts no threads except, if you ask, the step's own workers.
- **Proof, not trust.** Your program can write Elodin's deterministic witness format, so `elodin determinism compare` checks it against the Elodin run tick by tick and names the first byte that differs.

The library is a passive component. It does not require your class to inherit from an Elodin
type, and it does not own a scheduler, wall clock, files, callbacks, hooks, or processes. A
customer framework can contain an `elodin::Instance`, write its inputs, call one synchronous
`step`, and read its outputs whenever that framework's contract requires.

## What moves and what stays

```mermaid
flowchart LR
  sim["Simulation in Python"] -->|"World.to_jax and export_model"| model["Model directory"]
  model -->|"package.py"| pkg["ElodinModel package"]
  pkg -->|"find_package"| host["Your C++ program"]
  fsw["Your flight software"] --> host
  host --> witness["Witness"]
  elodin["The same simulation in Elodin"] --> base["Witness"]
  witness --> cmp["elodin determinism compare"]
  base --> cmp
```

| In Elodin | In your runtime |
|---|---|
| Systems, run by `World.run` | `libelodin_model`, stepped with `elodin_model_step` |
| `external_control` components a bridge or hook writes | `elodin_model_write` between steps |
| A `post_step` bridge or a `pre_commit` hook | Your exchange code |
| [s10 recipes](/reference/s10) that start and stop flight software | Your process supervision, or none if the flight software is linked in |
| Real-time pacing and the run receipt | Your scheduler and timers |
| Elodin-DB, the editor, sensor cameras | Not included |

The database and editor are not part of the library. To keep watching runs in the editor, have your program write to an Elodin-DB over [gRPC](/reference/grpc-client) or with the C++ headers from [`elodin-db gen-cpp`](/reference/elodin-cli#elodin-db-gen-cpp).

## Export the step

`World.to_jax` builds the world for JAX instead of running it, and `JaxSim.export_model` writes one tick of it to a directory. `World.export_model` is the direct form and can retain prepared batches as optional native interfaces:

```python
world, system, agents = build()
exchange = world.prepare_batch({
    "position": world.prepare_component("position", agents),
    "command": world.prepare_component("command", agents, access="write"),
})
world.export_model(
    system,
    "build/model",
    simulation_rate=RATE_HZ,
    workers=8,
    interfaces={"fsw.exchange": exchange},
    cpu="generic",
)
```

| File | Contents |
|---|---|
| `step.stablehlo.mlir` | One whole tick, from every component's value to its next value |
| `step.o` | The same tick compiled ahead of time by the Cranelift backend, for native or generic CPU features |
| `state.bin` | The initial value of every component, as raw little-endian bytes |
| `schema.json` | Model ABI, time and target metadata, typed components/entities, and optional named row interfaces |
| `model-manifest.json` | SHA-256 and size of every exported artifact |

One step is a whole tick, not just the systems: it advances `tick`, carries forward the components no system writes, and puts each output back in its component's place. Your program only swaps buffers.

**Match `workers` to the run you reproduce.** With more than one worker, Elodin traces each `@el.map` system in chunks of two entities so the chunks can run in parallel, and a chunked sum can round differently from a whole-column sum. `to_jax(workers=8)` traces the same way, so its step reproduces `World.run(workers=8)`; pass `workers=1` to reproduce a run without `workers`. Once the step is exported, the number of threads that runs it does not change its results.

## Package the library

```sh
python3 libs/elodin-model/package.py build/model build/ElodinModel
```

[`package.py`](https://github.com/elodin-sys/elodin/blob/main/libs/elodin-model/package.py) builds `libs/elodin-model` around the model with Cargo and lays out a CMake package. It needs a checkout of the Elodin repository and its pinned Rust toolchain; it is not part of the `elodin` pip package. Package with the same checkout that exported the model.

```text
ElodinModel/
  ElodinModelConfig.cmake
  include/elodin_model.h, elodin_model.hpp, elodin_step_hook.h
  lib/libelodin_model.so   (libelodin_model.dylib on macOS)
  lib/libelodin_model.a
  schema.json, state.bin, step.o, step.stablehlo.mlir
  model-manifest.json, build-manifest.json, runtime-sbom.json
```

`--backend` decides how the model's code gets into the library:

| | `aot` (default) | `jit` |
|---|---|---|
| What the library holds | `step.o`, linked in, and the small runtime its calls need | The StableHLO and Elodin's Cranelift compiler |
| When code is generated | When you export | Once, in `elodin_model_open` |
| Size, many-agents model | 1.9 MB | 6.2 MB |
| CPU | The exporting machine's architecture and instruction set | The machine that opens it |
| Worker threads | Yes | No; the step runs on the calling thread |

Both run the same Cranelift code and give identical results; the many-agents witness has the same root on either. Choose `aot` when nothing may compile at run time, `jit` when the program runs on a different CPU than the one that exported it.

## Build your program

[`cpp/CMakeLists.txt`](https://github.com/elodin-sys/elodin/blob/main/examples/ball-cpp/cpp/CMakeLists.txt) in ball-cpp needs two lines to use the model:

```cmake
find_package(ElodinModel CONFIG REQUIRED)
target_link_libraries(ball_cpp PRIVATE ElodinModel::model)
```

`ElodinModel::model` and `ElodinModel::shared` are the shared library;
`ElodinModel::static` supports a customer executable that must link the runtime directly.
`elodin_model.hpp` is a header-only RAII wrapper designed for composition, not inheritance.

Configure with `-DElodinModel_DIR=<package>`. The library's install name is `@rpath/libelodin_model.dylib` on macOS and its soname `libelodin_model.so` on Linux, so give the executable a relative runpath, `@loader_path/../lib` or `$ORIGIN/../lib`, and `install(IMPORTED_RUNTIME_ARTIFACTS ElodinModel::model)` copies the library next to it. The installed directory can then move.

The core of ball-cpp's `main.cpp`:

```cpp
ElodinModelProgram* program = elodin_model_open();
ElodinModelInstance* ball = elodin_model_create(program);

const size_t world_pos = elodin_model_component_index(program, "world_pos");
double pos[7];
for (long tick = 0; tick < ticks; ++tick) {
    elodin_model_step(ball);
    elodin_model_read(ball, world_pos, pos, sizeof pos);
}

elodin_model_destroy(ball);
elodin_model_close(program);
```

## The C interface

[`elodin_model.h`](https://github.com/elodin-sys/elodin/blob/main/libs/elodin-model/include/elodin_model.h) separates a **program**, the model's code and initial state, from **instances**, one run each. Instances of one program are independent, and a program must outlive its instances.

| Group | Functions |
|---|---|
| Lifecycle | `elodin_model_open`, `elodin_model_close`, `elodin_model_create`, `elodin_model_destroy`, `elodin_model_reset` |
| Model contract | `elodin_model_abi_version`, `_schema_abi`, `_build_info`, `_nominal_time_step`, `_tick_advanced_by_step` |
| Components | `elodin_model_component_count`, `_name`, `_size`, `_index`, `_type`, `_read_rows`, `_write_rows` |
| Entities | `elodin_model_component_rows`, `elodin_model_entity_name`, `elodin_model_entity_id` |
| Named interfaces | `elodin_model_interface_count`, `_name`, `_field`, `_read`, `_write` |
| Stepping | `elodin_model_step`, `elodin_model_read`, `elodin_model_write` |
| Threads | `elodin_model_configure_workers`, `_effective_workers`, `_recommended_stack` |
| Errors and build | `elodin_model_error`, `elodin_model_backend` |

- **Components** are addressed by index, in `schema.json` order. Each holds every entity's value as one little-endian array; read and write it whole, with exactly `elodin_model_component_size` bytes. A wrong size fails and copies nothing.
- **Rows** are entities. Row `r` of a component belongs to `elodin_model_entity_id(program, c, r)`, the same id the entity has in Elodin. Use it to find an agent's row instead of assuming an order.
- **Writes** take effect at the next step, like an `external_control` write in Elodin. Write commands after a step and before the next.
- **Named interfaces** are optional prepared batches exported by the model author. They gather selected entity rows under stable aliases and access directions; they do not run callbacks or choose when a caller uses them.
- **Time belongs to the caller.** The schema reports nominal dt and tick semantics. `elodin_model_step` advances one model transition synchronously; it does not sleep or read wall time. `elodin_model_set_time_step` changes the model's dt component without stepping.
- **Reset** restores the embedded initial state. A long-lived application can reset, write case parameters and initial conditions, and run another case without reopening the program.
- **Errors.** Functions return `NULL` or nonzero on failure, and `elodin_model_error()` says why. The message is per thread and lasts until that thread's next failure.
- **Threads.** Use each instance from one thread at a time. `elodin_model_set_workers(n)`, before the first `elodin_model_open`, sets how many threads, the caller included, run an `aot` step's parallel groups; 0 uses the count the model was exported for.

## Run the loop yourself

Elodin's tick loop is short, and your program owns it now. many-agents-cpp keeps Elodin's order, which is what makes byte-for-byte comparison possible:

1. Step the simulation.
2. Read what the flight software needs.
3. Exchange with the flight software, and write its commands.
4. Record what changed.
5. Wait for the next tick, when paced.

A command written at tick N is used by the step at tick N+1, as in Elodin. The heart of [many-agents-cpp's `main.cpp`](https://github.com/elodin-sys/elodin/blob/main/examples/many-agents-cpp/cpp/main.cpp), abridged:

```cpp
for (uint64_t tick = 0; tick < ticks; ++tick) {
    model.step();
    const Column position = column(c.position, 3), velocity = column(c.velocity, 3);
    // ... the other components the flight software reads

    for (size_t agent = 0; agent < agents; ++agent)
        write_state_packet(states, agent, tick, position, velocity /* ... */);
    exchange->run(states.data(), replies.data());   // in-process, or UDP to processes
    for (size_t agent = 0; agent < agents; ++agent)
        take_command(replies, agent, tick, command);  // checks magic, version and tick
    model.write(c.command, command);

    if (witness)
        record_tick(*witness, tick /* ... */);
    pacer.wait();   // sleep to the next 1 ms deadline when real time
}
```

Pacing is yours to choose. The example sleeps to fixed 1 ms deadlines, counts a tick that ends after its deadline as late, and restarts the schedule after a tick more than two periods late, as Elodin's pacing does.

## Flight software in your runtime

The flight software can live in the same process, called once per tick, or run as one process per agent over UDP, as Elodin's recipes run it. many-agents-cpp does both behind one interface, and its witness is the same either way.

**Test the C++ flight software inside Elodin first.** Give it the command-line and packet interface of the flight software it replaces, and it drops into the unchanged Elodin SITL:

```sh
ELODIN_MANY_AGENT_CONTROLLER_BIN=build/install/bin/many_agents_fsw \
  elodin run examples/many-agents/main.py
```

If that run's witness root equals the run with the original controllers, the C++ flight software decides exactly what the original decided, before anything else changes.

**Turn off floating-point contraction.** Compile with `-ffp-contract=off`. Otherwise clang on arm64 fuses `1.4 * position_error + 0.45 * velocity_error` into one fused multiply-add, which rounds once instead of twice. In many-agents-cpp, the first reply then differs from the Rust controller's in its last bit, and the comparison names that byte. Rust does not contract floating-point operations, and neither does Cranelift, the compiler behind Elodin's step.

## Prove it matches

Prove equivalence in steps, from the cheapest check to the strongest. many-agents-cpp's [`verify.py`](https://github.com/elodin-sys/elodin/blob/main/examples/many-agents-cpp/verify.py) runs every one:

| Check | What it proves |
|---|---|
| The C++ trajectories equal the values Elodin recorded | The step and the loop compute the same states |
| The C++ witness equals the Elodin witness | Every field, packet, reply and commit matches, every tick |
| A second C++ run is identical | The C++ program is deterministic |
| A run with another thread count is identical | Threads and scheduling do not change the results |
| An injected one-bit fault is found at its tick, record, entity and byte | The comparison finds real differences, and where they are |
| Elodin's hook, replayed from the C++ witness, writes the same bytes | Given the C++ run's inputs, Elodin's own code agrees with it |

Compare two witnesses with [`elodin determinism compare`](/reference/elodin-cli#elodin-determinism-compare):

```sh
elodin determinism compare elodin.witness cpp.witness
```

```text
identical: 600 ticks, root 72885022e2689c9a53c9fcadae417195f632599bdac6229bfa7c52b609fdb78a
```

and with the fault injected at tick 300:

```text
first mismatch
  tick: 300
  stage: after_hook:0
  hook: many_agent_step
  channel: controller.response
  entity: agent_001
  row: 1
  byte: 3
  left: 0x00
  right: 0x01
```

### Writing a witness from C++

A witness records the exact bytes of chosen fields and channels at each stage of each tick, and hashes them into a BLAKE3 tree whose root covers the run. The [Python API](/reference/python-api/#witness-files) gives the file format. many-agents-cpp's [`witness.cpp`](https://github.com/elodin-sys/elodin/blob/main/examples/many-agents-cpp/cpp/witness.cpp) is a writer in about 270 lines, with a portable BLAKE3 in [`blake3.cpp`](https://github.com/elodin-sys/elodin/blob/main/examples/many-agents-cpp/cpp/blake3.cpp). Record the Elodin baseline with `raw=True` first, and match its schema. Three parts are subtle:

- **The schema must be byte-identical.** Elodin hashes the compact JSON of the schema, fields in declaration order, and checks that digest when it opens the file. Write it exactly as Elodin does, and use the entity ids from `elodin_model_entity_id`.
- **The commit flags follow Elodin's database.** After each commit, a witness records whether each row was recorded, and Elodin records a row only when its bytes differ from the last value recorded for it. Keep the last recorded value of every row, starting from the initial state, and compare bytes.
- **Label hooks with Elodin's names.** Labels are not hashed, but `elodin determinism replay-hook` finds a native hook by the name, batch and partitions in its label. Name the stand-in for the hook after the hook it replaces.

An empty channel is 32 zero bytes, not the digest of no bytes, and external channels, such as whole replies with their service times, are kept for replay but never hashed.

### Keep the evidence

Keep the model directory, the Elodin and C++ witnesses, the `elodin determinism` output and the compiler flags with each delivery. Together they show that this program, built this way, reproduces this Elodin run on this machine.

{% alert(kind="notice") %}
A matching witness is evidence of equivalence between two runs. It is not tool qualification of Elodin, Cranelift, or your compiler under a standard such as DO-178C; plan for that separately if your process requires it.
{% end %}

## Threads and real time

Export with `workers` above 1 and an `aot` library runs the step's independent calls on a pool of threads it starts at open. These are many-agents-cpp runs with 24 agents, 20,000 ticks paced at 1 kHz, on an Apple M2 Max (macOS):

| Run | Step, mean / p95 | Exchange, mean / p95 | Late ticks |
|---|---|---|---|
| C++, 8 workers, flight software in-process | 270 / 355 µs | 3 / 8 µs | 1 |
| C++, 8 workers, 24 flight-software processes over UDP | 299 / 377 µs | 288 / 344 µs | 147 (0.7%) |
| C++, 1 worker | 1,409 / 1,451 µs | 2 / 4 µs | All: it cannot keep 1 kHz |
| Elodin `lockstep`, 8 workers, 24 controller processes | 270 / 334 µs | 281 / 320 µs | 0 |

The step is the same compiled code in both, so it costs the same. Calling the flight software in-process removes the UDP round trips, which take most of a tick when it runs as processes. The witness root is the same at 1 worker and at 8.

## Shipping it

- **`step.o` targets the machine that exported it**, its architecture, OS and instruction-set extensions. Export on a machine like the target, or ship the `jit` backend.
- **Build the library outside the Nix shell for delivery.** Built in Elodin's Nix shell, the macOS library links `libiconv` and `libintl` from the Nix store, and the Linux library has a Nix store runpath and Nix's glibc. Built with the system toolchain, the macOS library needs only `libSystem` and `libiconv` from `/usr/lib`, and the Linux library needs only `libc`, `libm`, `libgcc_s` and the loader. Build on the oldest Linux you support, since glibc is backward compatible only.
- **One model per process.** The library and its C symbols have fixed names.
- **The entity count is fixed** at export. Export again for another number of agents.
- **The step runs on the caller's stack.** The examples need nothing special, but a large model can need more stack than a default thread has; Elodin runs its own simulation thread and step workers with 256 MB stacks.
- **What is not reproduced:** the database, telemetry, the editor, sensor cameras, and history reads and deferred writes in hooks. many-agents-cpp's witness covers its `lockstep` mode: the step, the exchange and the commits.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `elodin_model_open` fails with "this library holds no model" | The library was built without `ELODIN_MODEL_DIR`. Build it with `package.py`. |
| Open fails with "this library has no runtime function" | The model was exported by a different Elodin than the library was built from. Export and package with one checkout. |
| `elodin_model_read` or `_write` returns nonzero | The size must equal `elodin_model_component_size` for that component. |
| Results differ from Elodin in the last bits | Check that `to_jax(workers=)` matches the Elodin run, that C++ is compiled with `-ffp-contract=off`, and that both ran on the same machine. The witness names the first byte that differs. |
| `elodin determinism compare` reports different schemas | Entity ids or order, batch fields or channels differ from the baseline; the error names the first difference. |
| `replay-hook` says the witness has no native hook | The hook's label needs the hook's name, its batch and a non-empty partition list. |
| The installed program cannot find the library | Give it a relative runpath and install the library with it; see [Build your program](#build-your-program). |
| The Linux library fails on another machine with a glibc version error | Build it outside the Nix shell, on the oldest distribution you support. |
| macOS `ld` warns "no platform load command found in step.o" | Harmless: Cranelift's object has no platform command, and the linker assumes macOS. |
| The program crashes inside the step on a worker thread of its own | Give that thread a larger stack. |

## Examples

| Example | Shows |
|---|---|
| [ball-cpp](https://github.com/elodin-sys/elodin/tree/main/examples/ball-cpp) | The whole path for one body: export, package with both backends, a CMake program, relocation, and a CSV that equals Elodin bit for bit |
| [many-agents-cpp](https://github.com/elodin-sys/elodin/tree/main/examples/many-agents-cpp) | A 24-agent SITL in C++: flight software in-process or as processes, worker threads, pacing, a witness writer, and every check on this page |

## Related docs

{% cardlink(title="SITL & HITL", icon="rocket", href="/reference/sitl-hitl") %}
Fly flight software against an Elodin simulation, and record a witness of the run.
{% end %}

{% cardlink(title="Python API: JaxSim", icon="book", href="/reference/python-api/#class-elodin-jaxsim-jaxsim") %}
`World.to_jax`, the pure step functions, and `export_model`.
{% end %}

{% cardlink(title="Python API: Deterministic witnesses", icon="book", href="/reference/python-api/#deterministic-witnesses") %}
What a witness records, how it is hashed, and the file format.
{% end %}

{% cardlink(title="Elodin CLI: elodin determinism", icon="cog", href="/reference/elodin-cli#elodin-determinism") %}
Compare witnesses and replay hooks from them.
{% end %}
