# Many-Agent Lockstep Scaling

This example is a public, synthetic performance contract for one Elodin world
running many independently controlled dynamic agents. It contains no external
data or assets.

The default workload contains:

- 24 controlled agents;
- 24 deterministic observed objects;
- a 1 kHz graph and recording cadence;
- one isolated compiled controller process per agent;
- one `pre_commit` hook that prepares all frames, sends them together, collects
  all responses, and applies commands in stable order: the Rust library in
  `hook/` by default, or the same exchange in Python.

The workload intentionally exercises graph execution, DB synchronization,
Python callback scaling, process supervision, and loopback lockstep exchange in
one deliverable.

## Run

From the repository root:

```sh
python examples/many-agent-lockstep/controller/build.py
python examples/many-agent-lockstep/hook/build.py

elodin run examples/many-agent-lockstep/main.py
```

Useful environment variables:

```text
ELODIN_MANY_AGENT_AGENTS=24
ELODIN_MANY_AGENT_OBJECTS=24
ELODIN_MANY_AGENT_MAX_TICKS=10000
ELODIN_MANY_AGENT_MODE=runtime|local|lockstep
ELODIN_MANY_AGENT_IO=native|prepared|batch|scalar
ELODIN_MANY_AGENT_WORKERS=8
ELODIN_MANY_AGENT_HOOK_WORKERS=2
ELODIN_MANY_AGENT_REALTIME=0|1
ELODIN_MANY_AGENT_PORT_BASE=13000
ELODIN_MANY_AGENT_TIMEOUT_S=0.25
ELODIN_MANY_AGENT_AUX_COMPONENTS=32
ELODIN_MANY_AGENT_COMPUTE_ROUNDS=64
ELODIN_DB_PATH=dbs/many-agent-lockstep
```

`runtime` isolates graph and DB costs. `local` adds the callback that computes
the commands in Python. `lockstep` adds the supervised controller processes and
socket barrier.

The IO mode selects how the callback reaches the world. `native` (default)
runs the exchange in `hook/`, an `el.NativeStepHook` over prepared batches;
`local` mode computes commands in Python as with `prepared`. `prepared` runs
the exchange in a Python `pre_commit` hook over the same batches. `batch` and
`scalar` keep the original `post_step` callback, which reads and writes through
the DB with `StepContext`. All four produce the same outputs.

`WORKERS` sets `world.run(workers=...)`, the threads that run the graph's
entity chunks. `HOOK_WORKERS` sets how many threads split the native hook's
rows; more threads send sooner but wake more controllers at once.

Controller response time is mostly process wakeup. On laptops, deep CPU idle
states (C6 and below) add hundreds of microseconds to every wakeup; hosts that
idle in shallow states, as cloud VMs usually do, do not pay it.

## Benchmark

A short current-machine matrix:

```sh
python examples/many-agent-lockstep/benchmark.py \
  --sizes 1,2,4,8,10,12,16,20,24 \
  --modes runtime,local,lockstep \
  --warmup-ticks 2000 \
  --measured-ticks 10000 \
  --repeats 3 \
  --output profile_output/many-agent-lockstep
```

Certification on an exclusive x86_64 Linux VM:

```sh
python examples/many-agent-lockstep/benchmark.py \
  --sizes 24 \
  --modes lockstep \
  --warmup-ticks 2000 \
  --measured-ticks 60000 \
  --repeats 5 \
  --require-pass \
  --output profile_output/many-agent-lockstep-acceptance
```

`acceptance.json` contains the machine-readable contract. A public reference
class is AWS `c7i.8xlarge`; a documented equivalent VM is acceptable.

Every run writes a raw Elodin summary, canonical output digest, log, and
versioned receipt. `matrix.json` collects the runs and `forecast.json` reports
current-machine phase slopes. Current-machine projections are diagnostic only;
certification requires direct measurements on the acceptance VM.

## Handoff archive

```sh
python examples/many-agent-lockstep/package.py \
  profile_output/many-agent-lockstep-baseline \
  profile_output/many-agent-lockstep-paced \
  profile_output/many-agent-lockstep-profile \
  --output profile_output/many-agent-lockstep-handoff.tar.gz
```

The packager strips machine-local paths and network addresses, excludes DB and
build directories, and writes a relative-path SHA-256 manifest. Programs can
provide an additional regular-expression deny list with `--deny-file`.

## Requested Elodin capabilities

[`FEATURE_REQUESTS.md`](./FEATURE_REQUESTS.md) describes the API and runtime
changes this example is intended to drive. The Elodin team is expected to
upgrade this same example, preserve deterministic outputs, and return
before/after receipts proving the acceptance configuration passes.
