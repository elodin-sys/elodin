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
- **Deterministic witnesses.** `world.prepare_witness` and `el.DeterminismTrace`
  record the exact bytes of the agents and of every controller exchange, tick
  by tick, so two runs can be compared byte for byte and the hooks replayed
  without the simulation. See [Proving a run repeats](#proving-a-run-repeats).

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

## Proving a run repeats

In `lockstep`, every run should end in the same state, bit for bit. `main.py`
prepares a witness that records, on every tick:

- every row of the `exchange` and `objects` batches, after the systems run;
- for each hook, the state packet it sent each controller
  (`controller.ingress`), the ten values it took from each reply
  (`controller.response`), and the commands it wrote;
- whether the commit recorded each of those rows.

Each reply also ends with the controller's service time, which describes the
host, not the simulation. So the whole reply goes in an external channel,
`controller.reply`, which a replay feeds back to the hook and no comparison
looks at. Both hooks fill the same channels, so they record the same witness.

Set `ELODIN_MANY_AGENT_WITNESS_PATH` to record one, or pass `--witness` to the
benchmark. The steps below run 24 agents in real time from the repository root.

1. Build the controller and the native hook:

   ```sh
   python examples/many-agents/controller/build.py
   python examples/many-agents/hook/build.py
   ```

2. Record a baseline. `--witness raw` keeps the exact bytes of every tick, about
   280 MB for these 5,500 ticks; with digests only, a witness takes about 1 KB
   per tick:

   ```sh
   python examples/many-agents/benchmark.py --sizes 24 --modes lockstep \
     --repeats 1 --warmup-ticks 500 --measured-ticks 5000 \
     --witness raw --output /tmp/many-agents-baseline
   ```

3. Flip one bit of agent 7's response at tick 4096, and compare the run with
   the baseline as it goes:

   ```sh
   ELODIN_MANY_AGENT_FAULT_MODE=bit-flip \
   ELODIN_MANY_AGENT_FAULT_TICK=4096 ELODIN_MANY_AGENT_FAULT_AGENT=7 \
   ELODIN_MANY_AGENT_FAULT_BYTE=3 ELODIN_MANY_AGENT_FAULT_MASK=1 \
   python examples/many-agents/benchmark.py --sizes 24 --modes lockstep \
     --repeats 1 --warmup-ticks 500 --measured-ticks 5000 \
     --witness --compare-witness /tmp/many-agents-baseline/lockstep-24-run-01/run.witness \
     --output /tmp/many-agents-faulted
   ```

   ```text
   lockstep 24 996.4 Hz 1001.7 us/cycle witness c9279114b79f9bc8 first mismatch at tick 4096
   ```

   The run keeps only digests, except for the 16 ticks on either side of the
   first mismatch, whose exact bytes it keeps. `run.json` has the witness's
   path, root and first mismatch.

4. Compare the two witnesses:

   ```sh
   elodin determinism compare \
     /tmp/many-agents-baseline/lockstep-24-run-01/run.witness \
     /tmp/many-agents-faulted/lockstep-24-run-01/run.witness --stop-at-first
   ```

   ```text
   first mismatch
     tick: 4096
     stage: after_hook:0
     hook: many_agent_step
     channel: controller.response
     entity: agent_007
     row: 7
     byte: 3
     left: 0x69
     right: 0x68
   ```

   Without `--stop-at-first`, it also counts the ticks that differ: here 1,404,
   every tick from 4096 on.

5. Replay only the hook, without the simulation or the controllers. On macOS,
   the library is `libmany_agent_hook.dylib`:

   ```sh
   elodin determinism replay-hook /tmp/many-agents-baseline/lockstep-24-run-01/run.witness \
     --library examples/many-agents/hook/build/release/libmany_agent_hook.so \
     --symbol many_agent_step
   ```

   ```text
   reproduced: 5500 ticks, 22000 records, 148896000 bytes; 0 ticks had digests only
   ```

   This takes about 2 seconds. The hook reads the recorded replies instead of
   talking to controllers, and every byte it writes matches. Replaying the
   faulted witness instead, without the fault settings, reports the same first
   mismatch at tick 4096 from the bytes the run kept. To replay through the
   Python hook, run `main.py` with `ELODIN_MANY_AGENT_REPLAY` set to the
   witness and `ELODIN_MANY_AGENT_HOOK=python`.

6. Run the completion race: agent 7's controller reads state from a helper
   thread without waiting for it to finish.

   ```sh
   python examples/many-agents/benchmark.py --sizes 24 --modes lockstep \
     --repeats 20 --warmup-ticks 500 --measured-ticks 5000 \
     --fault-mode completion-race --witness --assert-bitwise \
     --output /tmp/many-agents-race
   ```

   The first repeat keeps every tick's exact bytes, and each later one compares
   with it as it runs. They part where the race starts, at the response, after
   the same state was sent:

   ```text
   lockstep 24 996.3 Hz 1000.9 us/cycle witness b90791b181667b60
   lockstep 24 997.9 Hz 1000.2 us/cycle witness 13cc98853be15f5a first mismatch at tick 4096
   ...
   runs with 24 agents ended in different states
   lockstep runs with 24 agents recorded different witnesses
   repeat 2:
   first mismatch
     tick: 4096
     stage: after_hook:0
     hook: many_agent_step
     channel: controller.response
     entity: agent_007
     row: 7
     byte: 0
     left: 0xaf
     right: 0x09
   ```

   `controller.ingress` comes before `controller.response` in the witness, so
   the state sent at that tick was the same: the controller answered the same
   question differently.

7. Switch to the causal token: the controller waits until the helper has
   finished the tick it asked for.

   ```sh
   python examples/many-agents/benchmark.py --sizes 24 --modes lockstep \
     --repeats 20 --warmup-ticks 500 --measured-ticks 5000 \
     --fault-mode causal-token --witness --assert-bitwise \
     --output /tmp/many-agents-causal
   ```

   All 20 repeats record the same witness, and the benchmark passes:

   ```text
   lockstep 24 999.4 Hz 999.0 us/cycle witness 107a940ce10f74ea
   lockstep 24 999.1 Hz 999.1 us/cycle witness 107a940ce10f74ea
   ...
   ```

8. Check any configuration the same way: `--assert-bitwise` fails the benchmark
   unless every repeat of each mode and size records the same witness. The
   first repeat keeps every tick's exact bytes, about 610 MB for the default
   12,000 ticks, and later repeats compare with it as they run.

   ```sh
   python examples/many-agents/benchmark.py --sizes 24 --modes lockstep \
     --repeats 10 --witness --assert-bitwise --output /tmp/many-agents-bitwise
   ```

### In `lockstep-history`

Which camera frames exist, and so what `history_step` reads and defers, depends
on when the render server delivers them. `main.py` leaves that hook's stage out
of the witness in this mode, with
`stages=["after_systems", "after_hook:0", "after_hook:2", "after_commit"]`. The
rest stays exact: the camera only watches. A replay skips the hooks whose stage
the witness leaves out.

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
| `ELODIN_MANY_AGENT_WITNESS_PATH` | none | where to record the run's witness |
| `ELODIN_MANY_AGENT_WITNESS_RAW` | `0` | `1` keeps every tick's exact bytes, `START:STOP` those ticks' |
| `ELODIN_MANY_AGENT_WITNESS_REFERENCE` | none | a witness to compare with as the run goes |
| `ELODIN_MANY_AGENT_REPLAY` | none | replay the hooks from this witness instead of running |
| `ELODIN_MANY_AGENT_FAULT_MODE` | none | `bit-flip`, `completion-race` or `causal-token` |
| `ELODIN_MANY_AGENT_FAULT_TICK` | `4096` | the tick of the bit flip, or the first tick of the race |
| `ELODIN_MANY_AGENT_FAULT_AGENT` | `7` | the agent whose response or controller is affected |
| `ELODIN_MANY_AGENT_FAULT_BYTE` | `3` | the byte of the response the bit flip changes |
| `ELODIN_MANY_AGENT_FAULT_MASK` | `1` | the bits the bit flip changes |
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

### Fault modes

These are deliberate defects, for trying out the witness. Don't use them for
measurements.

- `bit-flip` XORs one byte of one agent's response at one tick, in both hooks,
  after the reply is recorded.
- `completion-race` gives one agent's controller a helper thread that updates
  its state on every tick from the fault tick on. The controller wakes the
  helper and adds whatever state it has reached once the helper starts, without
  waiting for it to finish, so the reply depends on scheduling.
- `causal-token` is the fix: the controller waits until the helper has finished
  the tick it asked for, so every run gets the same replies.

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

Witness options:

- `--witness` records each run's witness; `--witness raw` also keeps every
  tick's exact bytes, and `--witness START:STOP` those of a tick window.
- `--compare-witness PATH` compares each run with a witness as it runs.
- `--replay-witness` replays each run's hooks from its witness, which needs
  exact bytes, and fails if any byte differs.
- `--fault-mode` sets `ELODIN_MANY_AGENT_FAULT_MODE`.
- `--assert-bitwise` fails unless every repeat of each mode and size records
  the same witness. The first repeat keeps every tick's exact bytes, and later
  repeats compare with it as they run, so the benchmark prints the tick, record,
  entity and byte where a repeat first differed.

Each run gets its own folder with:

- `summary.json`: the run receipt from Elodin;
- `run.json`: the run's settings and main numbers, and its witness's path,
  root and first mismatch;
- `run.log`: the output of the run;
- `db/`: the run's database, about 0.8 GB for 5,500 ticks at 24 agents, so
  delete output folders you are done with;
- `pacing.csv`: one row per tick, with `--trace`;
- `run.witness`, with `--witness`, and `replay.json`, with `--replay-witness`.

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
| `main.py` | builds the world, adds the camera, prepares the handles and the witness, defines the hooks, and runs or replays |
| `sim.py` | the components, the camera rig and the two systems |
| `schematic.py` | the scene the editor and the camera show |
| `protocol.py` | the UDP packets of the Python hook |
| `hook/` | the native hook, in Rust |
| `controller/` | the controller process, in Rust |
| `benchmark.py` | runs the example over many sizes and saves the timings |
| `tests/` | checks the systems, the packets, the benchmark, the history mode and the witness |
