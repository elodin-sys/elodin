# Many-Agent Lockstep

This example runs one Elodin world with 24 agents and 24 moving objects at
1 kHz. Each agent follows one object, steered by its own controller running in
a separate process. On every tick, Elodin sends each controller the state of
its agent, waits for all of the answers, and applies the commands before the
tick is recorded.

Use it to learn how to keep a large world with outside software in the loop
inside a 1 ms tick.

## What it shows

- **Parallel systems.** `world.run(workers=8)` runs the `@el.map` systems of
  `sim.py` on 8 threads.
- **Prepared batches.** `world.prepare_component` and `world.prepare_batch`
  choose the component rows once, before the run. A hook then reads and writes
  those rows in place: no lookups by name, and no trip through the database.
- **`pre_commit` hooks.** They run after the systems and before the tick is
  recorded. The commands they write are recorded with the tick and used by the
  systems on the next tick.
- **Native hooks.** `el.NativeStepHook` loads the Rust library in `hook/` and
  calls it every tick without holding Python's global lock. It can split the
  agents across threads.
- **External control.** `sim.py` marks `command` with `external_control="true"`
  because its value comes from the controllers, not from the systems. The
  simulation never overwrites it.
- **Run receipts.** Set `ELODIN_SIM_SUMMARY_JSON` to save how long each phase of
  the tick took. The benchmark does this for every run.
- **History reads.** In `lockstep-history` mode, `world.prepare_message_history`
  and `world.prepare_component_history` let a hook read what the database has
  already recorded: a frame by its exact time, or a position at or before a
  time.
- **Deferred writes.** `world.prepare_timestamped_writer` and `ctx.defer_write`
  queue rows that Elodin appends just after the tick is recorded, in the order
  the hooks queued them.
- **One ordered hook list.** Each hook sees what the hooks before it wrote. If a
  hook fails, the hooks after it and the tick's recording do not run, and the
  receipt names the hook.

Two more speedups need no code. Elodin saves a component only when a system or
hook wrote a new value. And since the example has no `pre_step` or `post_step`,
a background thread writes each tick to the database. History reads need every
earlier tick in the database, so in `lockstep-history` mode the hooks of each
tick first wait until the background thread has written the last one. The
thread usually finishes while the simulation waits for real time and steps the
systems, so this costs little. `synchronous_commits=True` asks for the same
thing, for example when another program follows each tick. The simulation then
also waits whenever the disk stalls, so keep the database on fast storage.

## How one tick works

1. Elodin reads into the world anything other programs wrote to the database
   (`copy_db_to_world`).
2. The systems in `sim.py` step every agent and object (`world_run`).
3. The hooks run in order (`pre_commit`). The native hook sends each controller
   one state packet, waits for every reply, and writes the commands into the
   `command` rows. In `lockstep-history` mode, the history hook then checks the
   frame that is due and queues its evidence.
4. Elodin records the tick (`commit`).
5. When running in real time, Elodin waits until the next tick is due.

In `lockstep-history` mode, the hooks of the next tick first wait for this tick
to be in the database, and Elodin appends the queued evidence
(`deferred_writes`).

The names in parentheses are the phases in the run receipt.

## Run

From the repository root:

```sh
python examples/many-agent-lockstep/controller/build.py
python examples/many-agent-lockstep/hook/build.py

elodin run examples/many-agent-lockstep/main.py
```

Settings are environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ELODIN_MANY_AGENT_AGENTS` | `24` | number of agents |
| `ELODIN_MANY_AGENT_OBJECTS` | number of agents | number of moving objects |
| `ELODIN_MANY_AGENT_MAX_TICKS` | `10000` | ticks to run |
| `ELODIN_MANY_AGENT_MODE` | `lockstep` | `runtime`, `local`, `lockstep` or `lockstep-history` |
| `ELODIN_MANY_AGENT_HOOK` | `native` | `native` or `python` |
| `ELODIN_MANY_AGENT_WORKERS` | `8` | threads for the systems |
| `ELODIN_MANY_AGENT_HOOK_WORKERS` | `2` | threads for the native hook |
| `ELODIN_MANY_AGENT_REALTIME` | `1` | `1` keeps wall-clock time, `0` runs as fast as it can |
| `ELODIN_MANY_AGENT_PORT_BASE` | `13000` | first UDP port; each agent uses two |
| `ELODIN_MANY_AGENT_TIMEOUT_S` | `0.25` | how long to wait for a controller |
| `ELODIN_MANY_AGENT_AUX_COMPONENTS` | `32` | extra components on each entity |
| `ELODIN_MANY_AGENT_COMPUTE_ROUNDS` | `64` | extra math for each entity on every tick |
| `ELODIN_MANY_AGENT_PRODUCER_DELAY_MS` | `2` | how long the producer takes to make a frame |
| `ELODIN_MANY_AGENT_PRODUCER_DROP_EVERY` | `7` | the producer drops every 7th frame; `0` keeps them all |
| `ELODIN_MANY_AGENT_PRODUCER_FRAMES` | `0` | the producer stops after this many frames; `0` never stops |
| `ELODIN_MANY_AGENT_PRODUCER_TIMEOUT_S` | `10` | how long the history hook waits for the producer |
| `ELODIN_DB_PATH` | `dbs/many-agent-lockstep` | where the run is recorded |

The extra components and math give each entity the size and cost of a real
vehicle model.

### Modes

- `runtime` runs the systems and records the ticks. There are no hooks and no
  controllers, so it shows the base cost.
- `local` adds a Python hook that computes the commands itself, with the same
  math as the controllers.
- `lockstep` adds the controller processes. Elodin starts one for each agent
  and stops them at the end of the run.
- `lockstep-history` adds `producer.py`, a process that stands in for a camera.
  Every 10 ticks it reads what the database recorded for agent 0 and its
  object, waits 2 ms, and publishes a small frame stamped with that time. It
  drops every 7th frame on purpose. Ten ticks after each capture, the history
  hook in `main.py` waits until the producer has handled it, reads the frame by
  its exact time, checks it against the recorded positions at that time, and
  queues evidence: the frame number, its time, and whether it was found (0) or
  missing (1).

The evidence is the same on every run, and if the producer stops, the run
fails. A frame the hook had to wait for counts as late. That depends on timing,
so it is reported in the result but is not part of the evidence.

### Hooks

- `native` uses the Rust library in `hook/`. With `HOOK_WORKERS=2`, two threads
  each serve half of the agents. More threads send sooner, but wake more
  controllers at the same moment.
- `python` runs the same exchange in Python: `exchange_step` and
  `ControllerBridge` in `main.py`. The `local` mode always uses it.

Both hooks give the same results, bit for bit. Results also stay the same for
any `WORKERS` above 1; going from 1 thread to more can change the last bits of
some sums.

Most of a controller's response time is the time to wake its process. On
laptops, deep CPU sleep states (C6 and below) add hundreds of microseconds to
every wakeup. Cloud VMs usually avoid those states and do not pay this.

## Measure

`benchmark.py` builds the controller and hook, runs the example for each size
and mode, and saves the results:

```sh
python examples/many-agent-lockstep/benchmark.py \
  --sizes 1,2,4,8,16,24 \
  --modes runtime,local,lockstep,lockstep-history \
  --repeats 3 \
  --output profile_output/many-agent-lockstep
```

Each run warms up for 2,000 ticks, then measures 10,000. Other options are
`--hook`, `--workers`, `--hook-workers`, `--no-paced` to run as fast as
possible, and `--trace` to also save the timing of every tick.

Each run gets its own folder with:

- `summary.json`: the run receipt from Elodin;
- `run.json`: the run's settings and main numbers;
- `run.log`: the output of the run;
- `pacing.csv`: one row per tick, with `--trace`.

`matrix.json` lists every run. The main numbers are:

- `effective_hz`: ticks per second of wall time;
- `late_fraction`: the share of ticks that ended after their deadline;
- `pacing_resets`: how often the run fell far enough behind to restart its
  schedule;
- the mean, 95th percentile and worst time of each tick phase, in
  microseconds; `callback` adds up the hooks and the deferred writes;
- `history_queries`, `history_misses` and `deferred_rows`, from the history
  mode.

Compare `lockstep` with `lockstep-history` to see what the history hook costs.
The `local`, `lockstep` and `lockstep-history` runs of one size must end in the
same state, and every `lockstep-history` run must record the same evidence. If
they don't, the benchmark lists the size and fails.

## Files

| File | What it does |
| --- | --- |
| `main.py` | builds the world, prepares the batch, defines the hooks, and runs |
| `sim.py` | the components and the two systems |
| `protocol.py` | the packets of the Python hook, and the producer's frames |
| `producer.py` | the stand-in camera of `lockstep-history` mode |
| `hook/` | the native hook, in Rust |
| `controller/` | the controller process, in Rust |
| `benchmark.py` | runs the example over many sizes and saves the timings |
| `tests/` | checks the systems, the packets, the benchmark and the history mode |
