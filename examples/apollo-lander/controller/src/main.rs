use std::env;
use std::io;
use std::net::UdpSocket;
use std::time::Duration;

const STATE_F64S: usize = 15;
const COMMAND_F64S: usize = 6;
const MIN_THROTTLE: f64 = 4_670.0 / 45_040.0;
// Must exceed the reference profile's initial ~97 m/s range-rate so guidance
// can acquire the descent from above.
const MAX_DESCENT_RATE_MPS: f64 = 120.0;
// Terminal contact rate; the real LM touched down at roughly 0.5 m/s.
const MIN_DESCENT_RATE_MPS: f64 = 0.5;
const MIN_VERTICAL_ACCEL_MPS2: f64 = 0.05;
const MAX_TILT_DEG: f64 = 30.0;

#[derive(Clone, Copy)]
struct State {
    altitude: f64,
    vertical_speed: f64,
    world_vel: [f64; 3],
    mass: f64,
    ref_alt: f64,
    ref_rate: f64,
    gravity: f64,
    max_thrust: f64,
    thrust_scale: f64,
    track_gain: f64,
    vertical_gain: f64,
    horizontal_gain: f64,
}

fn port_env(name: &str, default: u16) -> u16 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_state(bytes: &[u8]) -> Option<State> {
    if bytes.len() < STATE_F64S * 8 {
        return None;
    }
    let mut values = [0.0_f64; STATE_F64S];
    for (idx, value) in values.iter_mut().enumerate() {
        let start = idx * 8;
        let mut raw = [0_u8; 8];
        raw.copy_from_slice(&bytes[start..start + 8]);
        *value = f64::from_le_bytes(raw);
    }
    Some(State {
        altitude: values[1],
        vertical_speed: values[2],
        world_vel: [values[3], values[4], values[5]],
        mass: values[6],
        ref_alt: values[7],
        ref_rate: values[8],
        gravity: values[9],
        max_thrust: values[10],
        thrust_scale: values[11],
        track_gain: values[12],
        vertical_gain: values[13],
        horizontal_gain: values[14],
    })
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-9 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn quat_from_body_z(direction: [f64; 3]) -> [f64; 4] {
    let d = normalize(direction);
    let cross = [-d[1], d[0], 0.0];
    let dot = d[2].clamp(-1.0, 1.0);
    if dot < -0.999_999 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let q = [cross[0], cross[1], cross[2], 1.0 + dot];
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn clamp_horizontal(ax: f64, ay: f64, vertical_accel: f64) -> (f64, f64) {
    let limit = vertical_accel.max(MIN_VERTICAL_ACCEL_MPS2) * MAX_TILT_DEG.to_radians().tan();
    let mag = ax.hypot(ay);
    if mag <= limit || mag < 1e-9 {
        (ax, ay)
    } else {
        let scale = limit / mag;
        (ax * scale, ay * scale)
    }
}

fn command(state: State) -> [f64; COMMAND_F64S] {
    let rate_cmd = (state.ref_rate + state.track_gain * (state.ref_alt - state.altitude))
        .clamp(-MAX_DESCENT_RATE_MPS, -MIN_DESCENT_RATE_MPS);
    let vertical_accel = (state.gravity + state.vertical_gain * (rate_cmd - state.vertical_speed))
        .max(MIN_VERTICAL_ACCEL_MPS2);
    let (ax, ay) = clamp_horizontal(
        -state.horizontal_gain * state.world_vel[0],
        -state.horizontal_gain * state.world_vel[1],
        vertical_accel,
    );
    let desired_acc = [ax, ay, vertical_accel];
    let thrust_required = state.mass * (ax * ax + ay * ay + vertical_accel * vertical_accel).sqrt();
    let throttle = (thrust_required / (state.max_thrust * state.thrust_scale).max(1.0))
        .clamp(MIN_THROTTLE, 1.0);
    let q = quat_from_body_z(desired_acc);
    [throttle, q[0], q[1], q[2], q[3], rate_cmd]
}

fn encode(values: [f64; COMMAND_F64S]) -> [u8; COMMAND_F64S * 8] {
    let mut out = [0_u8; COMMAND_F64S * 8];
    for (idx, value) in values.iter().enumerate() {
        out[idx * 8..idx * 8 + 8].copy_from_slice(&value.to_le_bytes());
    }
    out
}

fn main() -> io::Result<()> {
    let state_port = port_env("ELODIN_MONTE_CARLO_STATE_PORT", 9013);
    let command_port = port_env("ELODIN_MONTE_CARLO_COMMAND_PORT", 9012);
    let recv = UdpSocket::bind(("127.0.0.1", state_port))?;
    let send = UdpSocket::bind(("127.0.0.1", 0))?;
    recv.set_read_timeout(Some(Duration::from_millis(250)))?;
    let mut buf = [0_u8; STATE_F64S * 8];
    loop {
        match recv.recv_from(&mut buf) {
            Ok((n, _)) => {
                if let Some(state) = parse_state(&buf[..n]) {
                    let out = encode(command(state));
                    send.send_to(&out, ("127.0.0.1", command_port))?;
                }
            }
            Err(err)
                if matches!(
                    err.kind(),
                    io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::Interrupted
                ) => {}
            Err(err) => return Err(err),
        }
    }
}
