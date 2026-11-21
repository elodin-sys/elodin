# ElodinModel passive native model

`JaxSim.export_model` writes one synchronous Elodin state transition. This crate packages that
transition behind a C ABI and a header-only C++ wrapper. A consuming program owns when to call
it, wall-clock pacing, logical-time policy, external I/O, process lifecycle, and evidence policy.

The model library does not start a simulation loop, read configuration files, invoke hooks, open
a database, or call back into the consumer.

## Caller lifecycle

```c
elodin_model_configure_workers(1); /* before open; 0 uses the exported count */
ElodinModelProgram *program = elodin_model_open();
ElodinModelInstance *run = elodin_model_create(program);

elodin_model_reset_checked(run);
elodin_model_set_time_step(run, 0.001);
elodin_model_write(run, input, bytes, size);
elodin_model_step(run);
elodin_model_read(run, output, bytes, size);

elodin_model_destroy(run);
elodin_model_close(program);
```

Open one program and reset or create instances for successive cases in the same process. Reset
restores `state.bin`; the caller then writes case parameters and initial conditions again.

`elodin_model_step` advances exactly one model transition and returns synchronously on the
calling thread. The model's `tick` component advances if the exported graph contains it. The
caller decides whether that transition runs immediately, at a real-time deadline, or as part of
a batch.

## Versioned artifact

Model schema ABI 2 records:

- endianness, nominal rate and time step, tick semantics, export worker count and CPU policy;
- component dtype, shape, element shape, row bytes, entities, metadata, and whether the step
  writes it;
- optional named interfaces: aliases over selected component/entity rows with read or write
  access.

Old unversioned model directories still open as ABI 1.

`World.export_model(..., interfaces={"name": batch})` is the high-level export API. Interfaces
describe data only. They do not encode callback order, hooks, pacing, or an application class.

## CMake package

`package.py MODEL_DIR PACKAGE_DIR` produces:

- `ElodinModel::model` and `ElodinModel::shared`, the relocatable shared library;
- `ElodinModel::static`, the static library;
- `elodin_model.h`, `elodin_model.hpp`, and optional `elodin_step_hook.h`;
- the schema, state, StableHLO, AOT object, model/build manifests, and runtime SBOM.

The SBOM is a conservative normal-dependency closure with available license files. Optional
compiler code enabled in the current crate feature set can appear even when the linker removes it
from an AOT shared library; use the ELF dependency and symbol report as the delivered-binary view.

The default AOT backend generates no code at run time. Export with `cpu="generic"` for the
target architecture baseline, or `cpu="native"` for all features of the export host. A model
object remains architecture- and OS-specific.

For native RHEL 8 delivery, build the package with a glibc 2.28 system toolchain, outside the Nix
shell. `scripts/ci/elodin_model_rhel8.sh` builds and tests both link variants and rejects a Nix
runpath or newer GLIBC requirement.

## Optional native hooks

`elodin_step_hook.h` is independent of model stepping. Its adapter loads a hook library and lets
the caller invoke it with explicit columns, tick, and timestamp:

```c
ElodinHook *hook = elodin_hook_open("libhook.so", "exchange");
elodin_hook_initialize(hook, rows, columns, count);
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

- One thread at a time may use an instance. Separate instances may be stepped sequentially.
- The AOT worker pool is process-global and only grows. Configure it before the first open.
- Runtime worker count does not change results. Export worker mode can change tracing and bits.
- The calling thread needs the stack reported by `elodin_model_recommended_stack`.
- Components and entity counts are fixed at export.
- C pointers returned for names, shapes, and build information remain valid while the program is
  open. State values use copying APIs; no public zero-copy lifetime is implied.
