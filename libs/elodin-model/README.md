# ElodinModel passive native model

`JaxSim.export_model` writes one synchronous Elodin state transition. This crate packages that
transition behind a C ABI and a header-only C++ wrapper. A consuming program owns when to call
it, wall-clock pacing, logical-time policy, external I/O, process lifecycle, and evidence policy.

The model library does not start a simulation loop, read configuration files, invoke hooks, open
a database, or call back into the consumer.

## Caller lifecycle

A class of the caller's own holds the model as a member. There is no Elodin base class, and the
class decides when each call happens:

```cpp
#include "elodin_model.hpp"

class Interceptors {
public:
    // Many cases run in one process: reset, then write this case's inputs.
    void begin_case(const std::vector<double>& positions)
    {
        run_.reset();
        run_.write(position_, positions);
    }

    void update()
    {
        run_.write_interface(agents_, command_, commands_.data(), commands_.size());
        run_.step();
        run_.read(position_, positions_.data(), positions_.size());
    }

private:
    elodin::Program model_;                          // opens the model once
    elodin::Instance run_ { model_ };                // the state of one run
    std::size_t position_ = model_.component("position");
    std::size_t agents_ = model_.interface("agents");
    std::size_t command_ = model_.field(agents_, "command");
    std::vector<double> commands_ = std::vector<double>(3 * model_.rows(position_));
    std::vector<double> positions_ = commands_;
};
```

The same in C, for a program built without exceptions:

```c
elodin_model_set_workers(1); /* before open; 0 uses the exported count */
ElodinModelProgram *program = elodin_model_open();
ElodinModelInstance *run = elodin_model_create(program);

elodin_model_reset(run); /* the next case */
elodin_model_write(run, input, values, size);
elodin_model_step(run);
elodin_model_read(run, output, values, size);

elodin_model_destroy(run);
elodin_model_close(program);
```

`step` advances exactly one model transition and returns on the calling thread. The caller
decides whether that happens immediately, at a real-time deadline, or as part of a batch.

Open one program and reset its instances for successive cases in the same process. Reset
restores every component of `state.bin`; the caller then writes that case's parameters and
initial conditions again.

Two conveniences depend on what the model has, and fail with a message when it lacks it:

- `elodin_model_current_tick` needs a tick component, which a model has when a system reads
  the tick. Each step adds one to it.
- `elodin_model_set_time_step` needs a `simulation_time_step` component, which a model has
  when its systems read the step at run time, as `el.six_dof` does. Other systems keep the step
  they were built with.

`elodin_model_nominal_time_step` is the step the model starts with. Elodin keeps it in whole
nanoseconds, so a 120 Hz model reports 0.008333333 s, and writing 1/120 instead changes its
results.

## Versioned artifact

`schema.json`, model ABI 1, records the nominal time step, the tick and time-step components,
the export worker count, and for each component its dtype, shape, row bytes, entities, metadata
and whether a step writes it. Optional named interfaces alias chosen entity rows of components,
with read or write access. `model-manifest.json` records the target `step.o` was compiled for
and the hash of each file.

`World.export_model(..., interfaces={"name": batch})` is the high-level export API. Interfaces
describe data only. They do not encode callback order, hooks, pacing, or an application class.

## CMake package

```sh
python3 libs/elodin-model/package.py MODEL_DIR PACKAGE_DIR [--backend aot|jit] [--cpu native|generic]
```

produces:

- `ElodinModel::shared`, the relocatable shared library;
- `ElodinModel::static`, the static library;
- `elodin_model.h`, `elodin_model.hpp`, and `elodin_step_hook.h`;
- the schema, state, StableHLO, AOT object, model and build manifests, and runtime SBOM.

A program that does not use CMake links either library directly:

```sh
g++ -std=c++17 main.cpp -I PACKAGE_DIR/include PACKAGE_DIR/lib/libelodin_model.a -lpthread -ldl -lm
g++ -std=c++17 main.cpp -I PACKAGE_DIR/include -L PACKAGE_DIR/lib -lelodin_model -Wl,-rpath,PACKAGE_DIR/lib
```

The default AOT backend generates no code at run time. `step.o` is compiled for one OS and
architecture, and as exported for the exporting CPU's features. `--cpu` compiles it again on
the packaging machine from the model's StableHLO: `generic` for any CPU of that architecture,
`native` for that machine's own.

Packaging needs this repository, its pinned Rust toolchain and Python 3.11; exporting needs
the Elodin Python SDK. So a model exported on a workstation packages on the delivery machine:

```sh
# RHEL 8 with Rust and python3.11: no Elodin SDK, JAX or Nix
python3.11 libs/elodin-model/package.py model/ ElodinModel/ --cpu generic
```

Build there, outside the Nix shell, for a library that needs only the system's glibc.

The embedded build information, `elodin_model_build_info()`, is the JSON a run manifest needs:
compile time, Elodin revision, toolchain, target, and the hashes of the model's files. The SBOM
lists the library's normal dependencies for the packaging platform, with the license files
their sources ship. The compiler crates the `jit` backend uses are listed for `aot` too, although
the linker leaves them out of an `aot` shared library.

## Tests

`tests/fixtures/rooms` is a model small enough to check in: two rooms whose temperatures move
toward a commanded ambient temperature. The unit tests open it, and `tests/cpp` packages it
and checks the C interface, the C++ wrapper, the hook loader and concurrent instances against
known answers, with the compiler warnings a strict consumer enables:

```sh
bash scripts/ci/elodin_model_abi.sh                  # this machine, in nix develop
bash scripts/ci/elodin_model_abi.sh --backend jit
bash scripts/ci/elodin_model_rhel8.sh                # RHEL 8 userland in Docker, --cpu generic
```

The RHEL 8 script also rejects a library that needs a glibc newer than 2.28 or has a Nix
runpath. Two of the test programs work on any package: `threads` steps separate instances from
eight threads and compares them with the same runs alone, and `stack_probe` reports how much
stack the model's steps use.

## Optional native hooks

`elodin_step_hook.h` is independent of model stepping. It declares Elodin's native hook
interface for hook libraries, and a loader that lets the caller invoke one with explicit
columns, tick, and timestamp:

```c
ElodinHook *hook = elodin_hook_open("libhook.so", "exchange");
elodin_hook_initialize(hook, rows, tick, timestamp, columns, count);
elodin_hook_call(hook, tick, timestamp, columns, count);
elodin_hook_close(hook);
```

The caller decides where that call belongs. `elodin_model_step` never invokes it.

## Determinism

`elodin determinism inspect --json TRACE` validates a witness and prints its schema, labels,
trailer, and exact-byte inventory; `--schema PATH` extracts the validated schema for an outside
runtime. [WITNESS_C_API.md](WITNESS_C_API.md) defines the planned caller-driven C writer/reader.
It remains separate from model stepping so a customer orchestrator chooses its own evidence
stages and external channels.

## Constraints

- One thread at a time may use an instance. Different instances may be stepped from different
  threads at the same time; with the `jit` backend their steps take turns.
- The AOT worker threads are shared by the process and only grow in number. Configure them
  before the first open.
- Runtime worker count does not change results. Export worker mode can change tracing and bits.
- A step keeps its temporaries on the calling thread's stack. The examples use under 20 KiB;
  measure a model with `stack_probe`. `elodin_model_recommended_stack` is the 256 MiB Elodin
  gives its own threads, which fits any model.
- An error message lasts until the same thread's next failed call. Successful calls keep it.
- Components and entity counts are fixed at export, and a process holds one model.
- C pointers returned for names, shapes, and build information remain valid while the program is
  open. State values use copying APIs; no public zero-copy lifetime is implied.
