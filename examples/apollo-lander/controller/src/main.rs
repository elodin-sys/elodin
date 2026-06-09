use std::env;
use std::io;
use std::net::UdpSocket;
use std::time::Duration;

const STATE_F64S: usize = 20;
const COMMAND_F64S: usize = 6;
const MIN_THROTTLE: f64 = 4_670.0 / 45_040.0;
// The DPS fixed throttle point flown through the braking phase. The engine
// must not run between ~65% and FTP (nozzle erosion), so guidance commands
// above the band floor snap to FTP — the same logic LUMINARY used.
const FTP_THROTTLE: f64 = 0.925;
const EROSION_BAND_MIN: f64 = 0.65;
const MAX_DESCENT_RATE_MPS: f64 = 120.0;
// Terminal contact rate; the real LM touched down at roughly 0.5 m/s.
const MIN_DESCENT_RATE_MPS: f64 = 0.5;
const MIN_VERTICAL_ACCEL_MPS2: f64 = 0.05;
// Thrust-vector tilt budget: nearly horizontal while braking off orbital
// velocity, blending down to a tight cone around vertical for the approach
// and landing (the P64-style pitchover emerges from this blend).
const MAX_TILT_BRAKING_DEG: f64 = 82.0;
const MAX_TILT_APPROACH_DEG: f64 = 30.0;
const TILT_BLEND_HI_MPS: f64 = 150.0; // historical P63 -> P64 handoff speed
const TILT_BLEND_LO_MPS: f64 = 40.0;
// Horizontal trajectory-tracking gains (per-second feedback on the
// reconstructed downrange velocity / position profile). The position term is
// a slow trim with capped authority so initial-condition offsets are eroded
// over the long braking phase without destabilizing the velocity loop.
const HSPEED_GAIN: f64 = 0.25;
const POSITION_AUTHORITY_MPS2: f64 = 0.5;
// Bound on the altitude-tracking correction added to the reference descent
// rate, so altitude errors steer the profile without slewing the attitude.
const RATE_TRACK_AUTHORITY_MPS: f64 = 12.0;
// Feedback accelerations are bounded around the feed-forward profile: the
// reconstructed profile is flyable as-is, so corrections only need to erode
// small errors. Unbounded feedback lets the vertical and horizontal channels
// fight over the band-limited throttle and ring the attitude.
const VERTICAL_FB_AUTHORITY_MPS2: f64 = 0.8;
const HSPEED_FB_AUTHORITY_MPS2: f64 = 0.8;
// Below this altitude the controller stops following the reference profile
// and simply nulls all drift before letting down — P66 semantics: touchdown
// timing shifts with small altitude errors, and arriving early must not mean
// arriving with the reference's residual forward speed.
const TERMINAL_NULL_ALT_M: f64 = 40.0;
const R_MOON_M: f64 = 1_737_400.0;

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
    pos_x: f64,
    pos_y: f64,
    ref_downrange: f64,
    ref_hspeed: f64,
    ref_hdecel: f64,
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
        pos_x: values[15],
        pos_y: values[16],
        ref_downrange: values[17],
        ref_hspeed: values[18],
        ref_hdecel: values[19],
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

/// Cap the thrust-vector tilt while preserving the commanded magnitude: when
/// guidance asks for a nearly horizontal vector (braking while the altitude
/// loop wants to descend), rotate it up to the tilt limit instead of
/// collapsing the braking authority. Used at braking speeds.
fn cap_tilt_preserve_magnitude(ax: f64, ay: f64, az: f64, max_tilt_deg: f64) -> (f64, f64, f64) {
    let az = az.max(MIN_VERTICAL_ACCEL_MPS2);
    let ah = ax.hypot(ay);
    if ah < 1e-9 {
        return (ax, ay, az);
    }
    let max_tilt = max_tilt_deg.to_radians();
    if ah.atan2(az) <= max_tilt {
        return (ax, ay, az);
    }
    let mag = (ah * ah + az * az).sqrt();
    let scale_h = mag * max_tilt.sin() / ah;
    (ax * scale_h, ay * scale_h, mag * max_tilt.cos())
}

/// Shrink the horizontal acceleration into a tilt cone around "up",
/// preserving the vertical channel. Used at approach/landing speeds where a
/// soft touchdown takes priority over horizontal agility.
fn clamp_horizontal(ax: f64, ay: f64, vertical_accel: f64, max_tilt_deg: f64) -> (f64, f64) {
    let limit = vertical_accel.max(MIN_VERTICAL_ACCEL_MPS2) * max_tilt_deg.to_radians().tan();
    let mag = ax.hypot(ay);
    if mag <= limit || mag < 1e-9 {
        (ax, ay)
    } else {
        let scale = limit / mag;
        (ax * scale, ay * scale)
    }
}

/// Throttle state mirroring the LUMINARY throttle logic: the DPS holds the
/// fixed throttle point until guidance demand drops well below the erosion
/// band, then stays below the band (with a re-latch hysteresis kept for
/// off-nominal saves).
struct ThrottleLogic {
    ftp_latched: bool,
}

impl ThrottleLogic {
    fn apply(&mut self, demand: f64) -> f64 {
        if self.ftp_latched && demand < 0.60 {
            self.ftp_latched = false;
        } else if !self.ftp_latched && demand > 0.80 {
            self.ftp_latched = true;
        }
        if demand <= EROSION_BAND_MIN && !self.ftp_latched {
            demand.max(MIN_THROTTLE)
        } else if self.ftp_latched {
            FTP_THROTTLE
        } else {
            EROSION_BAND_MIN
        }
    }
}

fn command(state: State, throttle_logic: &mut ThrottleLogic) -> [f64; COMMAND_F64S] {
    let h_speed = state.world_vel[0].hypot(state.world_vel[1]);
    // Flat-world stand-in for the orbital centrifugal relief: at braking-phase
    // speeds the vehicle needs v^2/R less thrust to hold altitude.
    let g_eff = (state.gravity - h_speed * h_speed / R_MOON_M).max(0.05 * state.gravity);

    let rate_track = (state.track_gain * (state.ref_alt - state.altitude))
        .clamp(-RATE_TRACK_AUTHORITY_MPS, RATE_TRACK_AUTHORITY_MPS);
    let rate_cmd =
        (state.ref_rate + rate_track).clamp(-MAX_DESCENT_RATE_MPS, -MIN_DESCENT_RATE_MPS);
    let vertical_fb = (state.vertical_gain * (rate_cmd - state.vertical_speed))
        .clamp(-VERTICAL_FB_AUTHORITY_MPS2, VERTICAL_FB_AUTHORITY_MPS2);
    let vertical_accel = (g_eff + vertical_fb).max(MIN_VERTICAL_ACCEL_MPS2);

    // Horizontal trajectory tracking: feed-forward the reconstructed braking
    // deceleration, with feedback on the downrange velocity reference and a
    // capped position trim (the references decay to zero over the landing
    // site, so the same law brakes, performs the pitchover, and steers onto
    // the site). The trim fades out below ~150 m: like P66, the terminal
    // phase nulls drift and lands where it is rather than chasing position.
    let position_gain = 0.01 * state.horizontal_gain;
    let trim_fade = ((state.altitude - 30.0) / 120.0).clamp(0.0, 1.0);
    let trim_x = (position_gain * (state.ref_downrange - state.pos_x))
        .clamp(-POSITION_AUTHORITY_MPS2, POSITION_AUTHORITY_MPS2)
        * trim_fade;
    let trim_y = (position_gain * (-state.pos_y))
        .clamp(-POSITION_AUTHORITY_MPS2, POSITION_AUTHORITY_MPS2)
        * trim_fade;
    let (target_vx, target_decel) = if state.altitude < TERMINAL_NULL_ALT_M {
        (0.0, 0.0)
    } else {
        (state.ref_hspeed, state.ref_hdecel)
    };
    let hspeed_fb = (HSPEED_GAIN * (target_vx - state.world_vel[0]))
        .clamp(-HSPEED_FB_AUTHORITY_MPS2, HSPEED_FB_AUTHORITY_MPS2);
    let ax = -target_decel + hspeed_fb + trim_x;
    let ay = (HSPEED_GAIN * (-state.world_vel[1]))
        .clamp(-HSPEED_FB_AUTHORITY_MPS2, HSPEED_FB_AUTHORITY_MPS2)
        + trim_y;

    let blend =
        ((h_speed - TILT_BLEND_LO_MPS) / (TILT_BLEND_HI_MPS - TILT_BLEND_LO_MPS)).clamp(0.0, 1.0);
    let max_tilt_deg =
        MAX_TILT_APPROACH_DEG + (MAX_TILT_BRAKING_DEG - MAX_TILT_APPROACH_DEG) * blend;
    let (ax, ay, vertical_accel) = if h_speed > TILT_BLEND_LO_MPS {
        // Braking regime: keep the commanded magnitude, rotating the vector
        // up to the tilt limit if the altitude loop asks for a steep descent.
        cap_tilt_preserve_magnitude(ax, ay, vertical_accel, max_tilt_deg)
    } else {
        // Approach/landing regime: soft touchdown takes priority.
        let (ax, ay) = clamp_horizontal(ax, ay, vertical_accel, MAX_TILT_APPROACH_DEG);
        (ax, ay, vertical_accel)
    };

    let desired_acc = [ax, ay, vertical_accel];
    let thrust_required = state.mass * (ax * ax + ay * ay + vertical_accel * vertical_accel).sqrt();
    let demand = (thrust_required / (state.max_thrust * state.thrust_scale).max(1.0))
        .clamp(MIN_THROTTLE, FTP_THROTTLE);
    let throttle = throttle_logic.apply(demand);
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
    // The window opens mid-braking-burn with the DPS held at FTP.
    let mut throttle_logic = ThrottleLogic { ftp_latched: true };
    loop {
        match recv.recv_from(&mut buf) {
            Ok((n, _)) => {
                if let Some(state) = parse_state(&buf[..n]) {
                    let out = encode(command(state, &mut throttle_logic));
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
