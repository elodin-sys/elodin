# Many Agents

This example runs one Elodin world with 24 agents and 24 moving objects at
1 kHz. Each agent follows one object, steered by its own controller running in
a separate process. On every tick, Elodin sends each controller the state of
its agent, waits for all of the answers, and applies the commands before the
tick is recorded. A sensor camera hangs about 80 m above the scene and watches,
drifting slowly up and down so its view zooms in and out, and a hook reads its
frames back from the database by the time they were captured.

Use it to learn how to keep a large world with outside software in the loop
inside a 1 ms tick.

## Run

From the repository root:

```sh
python examples/many-agents/controller/build.py
python examples/many-agents/hook/build.py

elodin editor examples/many-agents/main.py
```

The left pane shows the scene: agents as blue spheres, the objects they follow
as orange spheres, and the yellow outline of what the camera sees. The right
pane shows the camera's own frames. To run without the editor, use
`elodin run examples/many-agents/main.py`.

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
- **Sensor cameras.** `world.sensor_camera` adds the overhead camera to the
  camera rig, which `bob_camera` in `sim.py` moves up and down by 12 m every
  20 seconds. `schematic.py` draws the scene that both the editor and the
  camera show.
- **History reads.** `world.prepare_message_history` and
  `world.prepare_component_history` let a hook read what the database has
  already recorded: a camera frame by its exact capture time, and an agent's
  position at that time.
- **Deferred writes.** `world.prepare_timestamped_writer` and `ctx.defer_write`
  queue rows that Elodin appends just after the tick is recorded, in the order
  the hooks queued them.
- **One ordered hook list.** Each hook sees what the hooks before it wrote. If a
  hook fails, the hooks after it and the tick's recording do not run, and the
  receipt names the hook.
- **Run receipts.** Set `ELODIN_SIM_SUMMARY_JSON` to save how long each phase of
  the tick took. The benchmark does this for every run.

Two more speedups need no code. Elodin saves a component only when a system or
hook wrote a new value. And a background thread writes each tick to the
database. History reads need every earlier tick in the database, so with the
camera the hooks of each tick first wait until the background thread has
written the last one. The thread usually finishes while the simulation waits
for real time and steps the systems, so this costs little.
`synchronous_commits=True` asks for the same thing, for example when another
program follows each tick. The simulation then also waits whenever the disk
stalls, so keep the database on fast storage.

## How one tick works

1. Elodin reads into the world anything other programs wrote to the database
   (`copy_db_to_world`).
2. The systems in `sim.py` step every agent and object (`world_run`).
3. The hooks run in order (`pre_commit`). The native hook sends each controller
   one state packet, waits for every reply, and writes the commands into the
   `command` rows. The history hook then checks the camera frame that is due
   and queues its evidence.
4. Elodin records the tick (`commit`).
5. When running in real time, Elodin waits until the next tick is due.

With the camera, the hooks of the next tick first wait for this tick to be in
the database, and Elodin appends the queued evidence (`deferred_writes`).

The names in parentheses are the phases in the run receipt.

## The camera and the history hook

Elodin starts its render server whenever the world has a sensor camera. The
server follows the recorded ticks, renders 25 frames per second of simulation
time, and records each frame in the database, stamped with the simulation time
it shows. Its first frame sets the schedule: the next ones follow every 40 ms.

A frame is due 80 ms after its capture time. Then `history_step` in `main.py`:

1. reads the frame by its exact capture time;
2. calls it found if it is there, and missing if a newer frame is there
   instead, because the render server skips frames to keep up. If neither is
   there, the frame is late, and the hook looks again on the next tick;
3. reads agent 0's position at the capture time;
4. queues the evidence: frame number, capture time, found (0) or missing (1),
   and that position.

If the camera sends nothing for 10 seconds, the run fails and the receipt names
`history_step`. Which frames exist depends on how fast the render server runs,
so the evidence differs a little from run to run. The simulation's own state
does not.

## Settings

Settings are environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ELODIN_MANY_AGENT_AGENTS` | `24` | number of agents |
| `ELODIN_MANY_AGENT_OBJECTS` | number of agents | number of moving objects |
| `ELODIN_MANY_AGENT_MAX_TICKS` | `60000` | ticks to run |
| `ELODIN_MANY_AGENT_MODE` | `lockstep-history` | `runtime`, `local`, `lockstep` or `lockstep-history` |
| `ELODIN_MANY_AGENT_HOOK` | `native` | `native` or `python` |
| `ELODIN_MANY_AGENT_WORKERS` | `8` | threads for the systems |
| `ELODIN_MANY_AGENT_HOOK_WORKERS` | `2` | threads for the native hook |
| `ELODIN_MANY_AGENT_REALTIME` | `1` | `1` keeps wall-clock time, `0` runs as fast as it can |
| `ELODIN_MANY_AGENT_PORT_BASE` | `13000` | first UDP port; each agent uses two |
| `ELODIN_MANY_AGENT_TIMEOUT_S` | `0.25` | how long to wait for a controller |
| `ELODIN_MANY_AGENT_CAMERA_TIMEOUT_S` | `10` | how long the history hook waits for camera frames |
| `ELODIN_MANY_AGENT_AUX_COMPONENTS` | `32` | extra components on each entity |
| `ELODIN_MANY_AGENT_COMPUTE_ROUNDS` | `64` | extra math for each entity on every tick |
| `ELODIN_DB_PATH` | `dbs/many-agents` | where the run is recorded |

The extra components and math give each entity the size and cost of a real
vehicle model.

### Modes

- `lockstep-history`, the default, runs the controller processes, the overhead
  camera and the history hook.
- `lockstep` runs the controller processes without the camera.
- `local` replaces the controllers with a Python hook that computes the
  commands itself, with the same math as the controllers.
- `runtime` runs the systems and records the ticks, with no hooks, so it shows
  the base cost.

Only `lockstep-history` has the camera, so the other modes measure the
simulation alone.

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
python examples/many-agents/benchmark.py \
  --sizes 1,2,4,8,16,24 \
  --modes runtime,local,lockstep,lockstep-history \
  --repeats 3 \
  --output profile_output/many-agents
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
  hook.

Compare `lockstep` with `lockstep-history` to see what the camera and the
history hook cost. The `local`, `lockstep` and `lockstep-history` runs of one
size must end in the same state. If they don't, the benchmark lists the size
and fails.

## Files

| File | What it does |
| --- | --- |
| `main.py` | builds the world, adds the camera, prepares the handles, defines the hooks, and runs |
| `sim.py` | the components, the camera rig and the two systems |
| `schematic.py` | the scene the editor and the camera show |
| `protocol.py` | the UDP packets of the Python hook |
| `hook/` | the native hook, in Rust |
| `controller/` | the controller process, in Rust |
| `benchmark.py` | runs the example over many sizes and saves the timings |
| `tests/` | checks the systems, the packets, the benchmark and the history mode |
