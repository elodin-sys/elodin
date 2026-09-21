# elodin.db Showcase — Fly a Crazyflie from Plain Python

This example demonstrates the standalone `elodin.db` client: a plain Python
process (no Elodin simulation) that starts an embedded Elodin DB, streams a
synthetic Crazyflie flight into it, and drives a live Editor session — the
`crazyflie.glb` model flies a figure-8 in a 3D viewport while graphs plot
every written signal.

## Run

From the repository root, inside the nix devshell (after `just install`):

```sh
uv run python examples/db-client/main.py
```

Close the Editor window to stop the demo; the script then prints a read-back
summary of everything it wrote. Headless variant (no Editor, fixed duration):

```sh
uv run python examples/db-client/main.py --no-editor --duration 5
```

## What it exercises

| `elodin.db` API | Where |
| --- | --- |
| `Server.start` (embedded DB) | `main()` — serves a temp directory on `127.0.0.1:2240` |
| `Client.connect` | `main()` |
| `table_writer` + `write_nowait` (drop-oldest) | 100 Hz state writer: `drone.world_pos` (`f64[7]`, labeled quaternion + xyz), `drone.imu.accel`, `drone.imu.gyro`, `drone.propeller_angle` (drives the rotor `animate` joints) |
| `table_writer` + blocking `write` | 10 Hz status writer: `drone.motor.rpm` (`f32[4]`), `drone.battery.voltage` (`f64`), `drone.status.armed` (`bool`), `drone.status.mode` (`i32`) |
| `Field` DSL (`f64[7].labeled(...)`, `f32`, `bool_`, `i32`) | writer schemas |
| `stream` (live) → derived write-back | `derived_loop` consumes `drone.world_pos` rows and publishes `drone.nav.speed` |
| `send_msg` (message log) | one `drone.events` JSON event per completed lap |
| `components`, `earliest_timestamp`, `latest`, `time_series`, `sql` + `sql_table_name`, `get_msgs` | shutdown summary |
| Writer observability (`dropped`, `state()`) | logged when the flight loop stops |

## Editor schematic

[`schematic.kdl`](schematic.kdl) shows:

- a chase-camera viewport tracking `drone.world_pos` with the `crazyflie.glb`
  model (from the repo `assets/` folder — the script sets `ELODIN_ASSETS_DIR`),
  its four rotors spun by `animate` joints driven from the RPM-integrated
  `drone.propeller_angle`, and a `line_3d` trajectory trail
- **Flight** tab: accelerometer, gyroscope, and the stream-derived ground
  speed
- **Status** tab: battery voltage, motor RPM, armed flag, flight mode
- **Pose** tab: the raw 7-element `world_pos` (labeled `q0..q3, x, y, z`)

The axis labels on the graphs come from the `.labeled(...)` element names set
by the writers.
