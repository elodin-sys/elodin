//! Falcon 9 booster flight software (WHITEPAPER 11).
//!
//! A standalone process: reads a sensor packet over UDP, runs estimation,
//! phase management, and guidance, and replies with actuator/valve commands.
//! It never sees truth state — only the simulated sensors (WHITEPAPER 12).

mod math;
mod profile;

use math::*;
use profile::AscentProfile;
use std::net::UdpSocket;

const STATE_FLOATS: usize = 49;
const CMD_FLOATS: usize = 27;
const N_ENGINES: usize = 9;
const N_VALVES: usize = 8;
/// Max attitude-setpoint slew (rad/s) — smooths AeroDescent→LandingBurn handoff.
/// Fast enough that a ~20° ignition step settles in <1 s (avoids a long AoA spike).
const ATT_SLEW_RADPS: f64 = 0.70;
/// Fin effectiveness priors matching aero.py (EST).
const FIN_S_M2: f64 = 1.5;
const FIN_CN_DELTA: f64 = 1.2;
const FIN_LEVER_M: f64 = 22.0;
const FIN_I_TRANS: f64 = 1.5e7; // kg·m² pitch/yaw inertia proxy at landing mass

// Valve indices (mirror sim.py).
const V_HE_LOX: usize = 0;
const V_HE_RP1: usize = 2;
const V_MAIN_LOX: usize = 4;
const V_MAIN_RP1: usize = 5;
const V_TEATEB: usize = 6;
const V_PURGE: usize = 7;

const THROTTLE_MIN: f64 = 0.57;
const T_VAC_PER_ENGINE: f64 = 829_000.0; // matches plant constants
const A_E_M2: f64 = 0.681;
/// ZEM/ZEV terminal guidance (Guo/Hawkins/Wie) — LandingBurn.
const ZEM_WAYPOINT_ALT_M: f64 = 150.0;
const ZEM_WAYPOINT_VDOWN_MPS: f64 = 25.0;
const ZEM_V_TD_MPS: f64 = 1.2;
const ZEM_TILT_CAP_RAD: f64 = 0.25;
const ZEM_COMMIT_ALT_M: f64 = 50.0;
const ZEM_COMMIT_TGO_S: f64 = 5.0;
const ZEM_A_LAND_TGO: f64 = 12.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    PadPress = 0,
    VerticalRise = 1,
    PitchKick = 2,
    GravityTurn = 3,
    Meco = 4,
    Flip = 5,
    Boostback = 6,
    Coast = 7,
    EntryBurn = 8,
    AeroDescent = 9,
    LandingBurn = 10,
    Touchdown = 11,
}

/// Sensor packet layout (sim -> FSW), little-endian f64.
struct SensorPacket {
    t: f64,
    imu_accel: V3,
    imu_gyro: V3,
    gps_pos: V3,
    gps_vel: V3,
    gps_count: f64,
    /// Terminal-descent refinement (Phase 7); -1 when invalid.
    #[allow(dead_code)]
    radar_range: f64,
    /// Tank/inlet monitoring (fault logic is a stretch goal).
    #[allow(dead_code)]
    pressures: [f64; 4],
    lox_kg: f64,
    rp1_kg: f64,
    params: GuidanceParams,
    landed: f64,
}

/// Campaign-tunable guidance parameters, sent each exchange (apollo pattern).
#[derive(Clone, Copy, Debug)]
struct GuidanceParams {
    kick_deg: f64,
    kick_start_s: f64,
    kick_ramp_s: f64,
    bucket_throttle: f64,
    bucket_q_on_pa: f64,
    /// Reserved for a predictor-based bucket exit (packet layout stability).
    #[allow(dead_code)]
    bucket_q_off_pa: f64,
    meco_speed_mps: f64,
    azimuth_deg: f64,
    boostback_overshoot_m: f64,
    entry_ignite_speed_mps: f64,
    entry_dv_mps: f64,
    landing_accel_margin: f64,
    ascent_throttle: f64,
    /// Flight-path angle from horizontal commanded at MECO speed (deg).
    meco_fpa_deg: f64,
    /// Pitch-program shape exponent in speed space.
    pitch_exp: f64,
    /// Entry-burn engine throttle (the recorded 26 m/s^2 mean decel implies
    /// a throttled three-engine burn).
    entry_throttle: f64,
    /// The hoverslam solve only arms below this altitude — above it the
    /// aerodynamic descent does the braking (WHITEPAPER 11.4-11.5).
    landing_arm_alt_m: f64,
    /// Entry-burn ignition altitude gate (recorded: ~49.8 km).
    entry_ignite_alt_m: f64,
    /// FSW onboard aero database: engines-first CA * S_ref (m^2) used by the
    /// impact-point predictor. A real navigator carries a calibrated drag
    /// model; the campaign calibrates ours.
    fsw_cd_s_m2: f64,
    /// Boostback three-engine throttle (the recorded burn decelerates at
    /// ~27 m/s^2 — a throttled burn, not full thrust).
    boostback_throttle: f64,
    /// Fin attitude-loop natural frequency (rad/s); torque is divided by
    /// q̄·effectiveness so closed-loop bandwidth is q̄-invariant.
    fin_wn: f64,
    /// Landing-burn lateral approach-speed cap (m/s); retained for MC packet
    /// layout (ZEM/ZEV no longer reads it).
    #[allow(dead_code)]
    divert_speed_cap: f64,
    /// Aero-descent cross-track tilt cap (rad) at low q̄; scheduled down as q̄ rises.
    steer_tilt_cap: f64,
}

impl SensorPacket {
    fn parse(v: &[f64; STATE_FLOATS]) -> Self {
        SensorPacket {
            t: v[0],
            imu_accel: [v[1], v[2], v[3]],
            imu_gyro: [v[4], v[5], v[6]],
            gps_pos: [v[7], v[8], v[9]],
            gps_vel: [v[10], v[11], v[12]],
            gps_count: v[13],
            radar_range: v[14],
            pressures: [v[16], v[17], v[18], v[19]],
            lox_kg: v[20],
            rp1_kg: v[21],
            params: GuidanceParams {
                kick_deg: v[22],
                kick_start_s: v[23],
                kick_ramp_s: v[24],
                bucket_throttle: v[25],
                bucket_q_on_pa: v[26],
                bucket_q_off_pa: v[27],
                meco_speed_mps: v[28],
                azimuth_deg: v[29],
                boostback_overshoot_m: v[30],
                entry_ignite_speed_mps: v[31],
                entry_dv_mps: v[32],
                landing_accel_margin: v[33],
                ascent_throttle: v[34],
                meco_fpa_deg: v[35],
                pitch_exp: v[36],
                entry_throttle: v[37],
                landing_arm_alt_m: v[38],
                entry_ignite_alt_m: v[39],
                fsw_cd_s_m2: v[40],
                boostback_throttle: v[41],
                fin_wn: if v[46] > 0.1 { v[46] } else { 1.4 },
                divert_speed_cap: if v[47] > 1.0 { v[47] } else { 35.0 },
                steer_tilt_cap: if v[48] > 0.01 { v[48] } else { 0.15 },
            },
            landed: v[43],
        }
    }
}

#[derive(Default)]
struct Command {
    engines: [f64; N_ENGINES],
    valves: [f64; N_VALVES],
    attitude: Quat,
    tvc_enable: f64,
    rcs_enable: f64,
    fins: [f64; 3],
    phase: f64,
}

impl Command {
    fn pack(&self) -> [f64; CMD_FLOATS] {
        let mut out = [0.0; CMD_FLOATS];
        out[..9].copy_from_slice(&self.engines);
        out[9..17].copy_from_slice(&self.valves);
        out[17..21].copy_from_slice(&self.attitude);
        out[21] = self.tvc_enable;
        out[22] = self.rcs_enable;
        out[23..26].copy_from_slice(&self.fins);
        out[26] = self.phase;
        out
    }
}

/// IMU + GPS navigator with the rotating-frame mechanization (WHITEPAPER 12).
struct Navigator {
    pos: V3,
    vel: V3,
    att: Quat, // body -> ECEF
    last_gps_count: f64,
    last_t: f64,
    initialized: bool,
    /// Steady wind estimate (ECEF m/s) from GPS–IMU innovation.
    wind_est: V3,
    /// Radar-smoothed altitude (m); <0 when invalid.
    radar_alt_m: f64,
}

impl Navigator {
    fn new() -> Self {
        Navigator {
            pos: [0.0; 3],
            vel: [0.0; 3],
            att: QUAT_IDENT,
            last_gps_count: 0.0,
            last_t: 0.0,
            initialized: false,
            wind_est: [0.0; 3],
            radar_alt_m: -1.0,
        }
    }

    fn init(&mut self, s: &SensorPacket) {
        // Known pad state: GPS position, zero ECEF velocity, body +X up.
        self.pos = s.gps_pos;
        self.vel = [0.0; 3];
        let (lat, lon, _) = ecef_to_geodetic(self.pos);
        let up = scale(ned_basis(lat, lon)[2], -1.0);
        self.att = quat_between([1.0, 0.0, 0.0], up);
        self.last_gps_count = s.gps_count;
        self.last_t = s.t;
        self.initialized = true;
        self.wind_est = [0.0; 3];
        self.radar_alt_m = -1.0;
    }

    fn step(&mut self, s: &SensorPacket) {
        if !self.initialized {
            if s.gps_count > 0.0 {
                self.init(s);
            }
            return;
        }
        let dt = (s.t - self.last_t).clamp(0.0, 0.1);
        self.last_t = s.t;
        // Attitude: integrate the gyro, removing the Earth-rate term the
        // gyro measures but the ECEF-referenced attitude must not integrate.
        let omega_e_body = quat_rotate_inv(self.att, [0.0, 0.0, OMEGA_EARTH]);
        let omega_frame = sub(s.imu_gyro, omega_e_body);
        self.att = quat_integrate(self.att, omega_frame, dt);
        // Translation: specific force + gravity + fictitious terms.
        let f_e = quat_rotate(self.att, s.imu_accel);
        let a = add(add(f_e, gravity(self.pos)), frame_accel(self.pos, self.vel));
        self.vel = add(self.vel, scale(a, dt));
        self.pos = add(self.pos, scale(self.vel, dt));
        // Complementary GPS blend (no snap) — kills metre-class steps into
        // terminal guidance. Position gain ~0.2/fix, velocity ~0.5/fix.
        if s.gps_count > self.last_gps_count {
            let innov_v = sub(s.gps_vel, self.vel);
            self.wind_est = add(scale(self.wind_est, 0.95), scale(innov_v, 0.05));
            self.pos = add(self.pos, scale(sub(s.gps_pos, self.pos), 0.20));
            self.vel = add(self.vel, scale(sub(s.gps_vel, self.vel), 0.50));
            self.last_gps_count = s.gps_count;
        }
        // Radar altimeter below 500 m for h / t_go (when valid).
        if s.radar_range >= 0.0 && s.radar_range < 500.0 {
            let geo_alt = ecef_to_geodetic(self.pos).2;
            if self.radar_alt_m < 0.0 {
                self.radar_alt_m = s.radar_range;
            } else {
                self.radar_alt_m = 0.7 * self.radar_alt_m + 0.3 * s.radar_range;
            }
            let (lat, lon, _) = ecef_to_geodetic(self.pos);
            let up = scale(ned_basis(lat, lon)[2], -1.0);
            let dh = self.radar_alt_m - geo_alt;
            if dh.abs() < 50.0 {
                self.pos = add(self.pos, scale(up, 0.35 * dh));
            }
        } else {
            self.radar_alt_m = -1.0;
        }
    }

    fn altitude(&self) -> f64 {
        if self.radar_alt_m >= 0.0 {
            self.radar_alt_m
        } else {
            ecef_to_geodetic(self.pos).2
        }
    }

    fn speed(&self) -> f64 {
        norm(self.vel)
    }
}

struct Fsw {
    nav: Navigator,
    phase: Phase,
    phase_t0: f64,
    meco_speed_reached: bool,
    purge_until: f64,
    entry_v0: f64,
    pad_pos: V3,
    lz1_pos: V3,
    up_pad: V3,
    track_dir: V3,
    /// 1-3-1 landing-burn engine latch: (escalated to 3, back down to 1).
    landing_escalated: bool,
    landing_deescalated: bool,
    /// Commit-to-vertical latch (freeze lateral ZEM below ~50 m / short t_go).
    landing_vertical_commit: bool,
    /// The designed ascent reference: the recorded webcast profile.
    ascent_profile: Option<AscentProfile>,
    t_liftoff: f64,
    /// Previous attitude command (for slew limiting).
    att_cmd: Quat,
    att_cmd_init: bool,
}

impl Fsw {
    fn new() -> Self {
        let lz1 = geodetic_to_ecef(28.48580_f64.to_radians(), -80.54440_f64.to_radians(), 5.0);
        Fsw {
            nav: Navigator::new(),
            phase: Phase::PadPress,
            phase_t0: 0.0,
            meco_speed_reached: false,
            purge_until: -1.0,
            entry_v0: 0.0,
            pad_pos: [0.0; 3],
            lz1_pos: lz1,
            up_pad: [0.0, 0.0, 1.0],
            track_dir: [1.0, 0.0, 0.0],
            landing_escalated: false,
            landing_deescalated: false,
            landing_vertical_commit: false,
            ascent_profile: std::env::var("ELODIN_F9_PROFILE")
                .ok()
                .and_then(|p| AscentProfile::load(&p)),
            t_liftoff: -1.0,
            att_cmd: QUAT_IDENT,
            att_cmd_init: false,
        }
    }

    /// Slew-limit the attitude setpoint toward `desired` (max ATT_SLEW_RADPS).
    fn slew_attitude(&mut self, desired: Quat, dt: f64) -> Quat {
        if !self.att_cmd_init {
            self.att_cmd = desired;
            self.att_cmd_init = true;
            return desired;
        }
        // Angle of the relative quaternion q_err = conj(prev) * desired.
        let q_err = quat_mul(quat_conj(self.att_cmd), desired);
        let (axis, angle) = quat_to_axis_angle(q_err);
        let max_step = ATT_SLEW_RADPS * dt.max(1e-3);
        if angle <= max_step {
            self.att_cmd = desired;
        } else {
            self.att_cmd = quat_mul(self.att_cmd, quat_from_axis_angle(axis, max_step));
            self.att_cmd = quat_normalize(self.att_cmd);
        }
        self.att_cmd
    }

    fn set_phase(&mut self, p: Phase, t: f64) {
        if p != self.phase {
            eprintln!("[fsw] t={t:8.2}s  {:?} -> {p:?}", self.phase);
            self.phase = p;
            self.phase_t0 = t;
        }
    }

    fn cutoff_with_purge(&mut self, cmd: &mut Command, t: f64) {
        cmd.engines = [0.0; N_ENGINES];
        self.purge_until = t + 5.0;
    }

    fn step(&mut self, s: &SensorPacket) -> Command {
        self.nav.step(s);
        let mut cmd = Command {
            attitude: self.nav.att,
            phase: self.phase as i32 as f64,
            ..Default::default()
        };
        // Helium pressurization runs the whole flight; purge per schedule.
        cmd.valves[V_HE_LOX] = 1.0;
        cmd.valves[V_HE_RP1] = 1.0;
        cmd.valves[V_PURGE] = if s.t < self.purge_until { 1.0 } else { 0.0 };

        if !self.nav.initialized {
            return cmd;
        }
        let p = &s.params;
        let t = s.t;
        let alt = self.nav.altitude();
        let speed = self.nav.speed();

        // Common frames.
        if self.phase == Phase::PadPress && self.pad_pos == [0.0; 3] {
            self.pad_pos = self.nav.pos;
            let (lat, lon, _) = ecef_to_geodetic(self.pad_pos);
            let ned = ned_basis(lat, lon);
            self.up_pad = scale(ned[2], -1.0);
            let az = p.azimuth_deg.to_radians();
            self.track_dir = normalize(add(scale(ned[0], az.cos()), scale(ned[1], az.sin())));
        }
        let up_here = {
            let (lat, lon, _) = ecef_to_geodetic(self.nav.pos);
            scale(ned_basis(lat, lon)[2], -1.0)
        };
        // Liftoff mark for the profile clock: first sustained climb.
        if self.t_liftoff < 0.0 && dot(self.nav.vel, up_here) > 1.0 {
            self.t_liftoff = t;
        }

        match self.phase {
            Phase::PadPress => {
                // Open feed + igniter isolation, then light the nine.
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.attitude = quat_between([1.0, 0.0, 0.0], self.up_pad);
                cmd.tvc_enable = 1.0;
                if t >= 0.2 {
                    cmd.engines = [p.ascent_throttle; N_ENGINES];
                    self.set_phase(Phase::VerticalRise, t);
                }
            }
            Phase::VerticalRise => {
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.engines = [p.ascent_throttle; N_ENGINES];
                cmd.attitude = quat_between([1.0, 0.0, 0.0], self.up_pad);
                cmd.tvc_enable = 1.0;
                // No RCS during powered ascent: TVC owns pitch/yaw and the
                // symmetric stack has no roll disturbance — cold gas is
                // reserved for the flip/coast/descent (budget realism).
                if t >= p.kick_start_s {
                    self.set_phase(Phase::PitchKick, t);
                }
            }
            Phase::PitchKick => {
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.engines = [p.ascent_throttle; N_ENGINES];
                cmd.tvc_enable = 1.0;
                // Ramp the nose from vertical toward the track azimuth.
                let f = ((t - self.phase_t0) / p.kick_ramp_s).clamp(0.0, 1.0);
                let angle = f * p.kick_deg.to_radians();
                let dir = normalize(add(
                    scale(self.up_pad, angle.cos()),
                    scale(self.track_dir, angle.sin()),
                ));
                cmd.attitude = quat_between([1.0, 0.0, 0.0], dir);
                if f >= 1.0 && speed > 80.0 {
                    self.set_phase(Phase::GravityTurn, t);
                }
            }
            Phase::GravityTurn => {
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.tvc_enable = 1.0;
                // Lofted pitch program (WHITEPAPER 11.1): command the
                // flight-path angle as a function of speed, from vertical at
                // the kick down to `meco_fpa_deg` at MECO speed. RTLS flights
                // fly deliberately lofted; the exponent shapes the loft and
                // the campaign calibrates both.
                let v0 = 90.0;
                let f = ((speed - v0) / (p.meco_speed_mps - v0)).clamp(0.0, 1.0);
                let gamma = (90.0 - (90.0 - p.meco_fpa_deg) * f.powf(p.pitch_exp)).to_radians();
                let dir = normalize(add(
                    scale(up_here, gamma.sin()),
                    scale(self.track_dir, gamma.cos()),
                ));
                // Profile-following ascent (WHITEPAPER 11.1): the recorded
                // flight IS the designed reference. Feed-forward its
                // flight-path angle; close speed with bounded throttle
                // feedback (the monte-carlo skill's tracking pattern). The
                // parametric program is the fallback without a profile.
                let mut dir_cmd = dir;
                let mut u = p.ascent_throttle;
                if let (Some(profile), true) = (&self.ascent_profile, self.t_liftoff >= 0.0) {
                    let t_ref = t - self.t_liftoff;
                    let v_ref = profile.speed(t_ref);
                    let gamma_ref = (profile.vspeed(t_ref) / v_ref.max(30.0))
                        .clamp(-1.0, 1.0)
                        .asin();
                    // Altitude-error trim rotates the commanded path angle.
                    let alt_err = profile.altitude(t_ref) - alt;
                    let gamma_cmd =
                        (gamma_ref + (alt_err * 2.0e-4).clamp(-0.12, 0.12)).clamp(0.0, 1.55);
                    dir_cmd = normalize(add(
                        scale(up_here, gamma_cmd.sin()),
                        scale(self.track_dir, gamma_cmd.cos()),
                    ));
                    u = (p.ascent_throttle + (v_ref - speed) * 2.0e-3).clamp(0.62, 1.0);
                }
                cmd.attitude = quat_between([1.0, 0.0, 0.0], dir_cmd);
                // Throttle bucket floor still applies through Max-Q.
                let qbar = 0.5 * density(alt) * speed * speed;
                if qbar > p.bucket_q_on_pa && speed < 500.0 {
                    u = u.min(p.bucket_throttle);
                }
                // Acceleration limiting toward MECO (the real vehicle
                // throttles down to hold ~3.6 g as the stack lightens).
                let a_meas = norm(s.imu_accel);
                if a_meas > 34.0 {
                    u = (u * 34.0 / a_meas).max(THROTTLE_MIN);
                }
                cmd.engines = [u; N_ENGINES];
                if speed >= p.meco_speed_mps {
                    self.meco_speed_reached = true;
                    self.cutoff_with_purge(&mut cmd, t);
                    self.set_phase(Phase::Meco, t);
                }
            }
            Phase::Meco => {
                // Keep mains open a beat for shutdown, then close; stage sep
                // happens plant-side on the separation command timing.
                cmd.rcs_enable = 1.0;
                cmd.attitude = quat_between([1.0, 0.0, 0.0], normalize(self.nav.vel));
                if t - self.phase_t0 > 3.0 {
                    self.set_phase(Phase::Flip, t);
                }
            }
            Phase::Flip => {
                cmd.rcs_enable = 1.0;
                // Flip toward the boostback burn direction: where the IIP
                // needs to move, horizontally (the dog-leg heading).
                let (burn_dir, _) = self.boostback_solution(s, up_here, p);
                cmd.attitude = quat_between([1.0, 0.0, 0.0], burn_dir);
                let x_body = quat_rotate(self.nav.att, [1.0, 0.0, 0.0]);
                if dot(x_body, burn_dir) > 0.95 {
                    self.set_phase(Phase::Boostback, t);
                }
            }
            Phase::Boostback => {
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.tvc_enable = 1.0;
                cmd.rcs_enable = 1.0;
                // Full-vector IIP targeting (WHITEPAPER 11.3): LZ-1 is NOT on
                // the ascent track line (it sits ~14 km cross-track of it),
                // so the boostback flies a dog-leg — thrust along the
                // horizontal direction that walks the predicted impact point
                // onto the target. The entry burn will pull the IIP back
                // along the course by roughly `boostback_overshoot_m`, so
                // aim that far beyond LZ-1 along the current course.
                let (burn_dir, miss_h_mag) = self.boostback_solution(s, up_here, p);
                cmd.attitude = quat_between([1.0, 0.0, 0.0], burn_dir);
                let u = p.boostback_throttle.clamp(THROTTLE_MIN, 1.0);
                let mut engines = [0.0; N_ENGINES];
                engines[0] = u;
                engines[1] = u;
                engines[2] = u;
                cmd.engines = engines;
                if miss_h_mag < 1_000.0 {
                    eprintln!("[fsw] boostback cutoff: horizontal iip-to-target {miss_h_mag:.0} m");
                    self.cutoff_with_purge(&mut cmd, t);
                    self.set_phase(Phase::Coast, t);
                }
            }
            Phase::Coast => {
                cmd.rcs_enable = 1.0;
                // Engines-first for entry: point +X against the velocity.
                cmd.attitude = quat_between([1.0, 0.0, 0.0], scale(normalize(self.nav.vel), -1.0));
                let descending = dot(self.nav.vel, up_here) < 0.0;
                if descending && (speed >= p.entry_ignite_speed_mps || alt <= p.entry_ignite_alt_m)
                {
                    self.entry_v0 = speed;
                    self.set_phase(Phase::EntryBurn, t);
                }
            }
            Phase::EntryBurn => {
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.tvc_enable = 1.0;
                cmd.rcs_enable = 1.0;
                cmd.attitude = quat_between([1.0, 0.0, 0.0], scale(normalize(self.nav.vel), -1.0));
                let mut engines = [0.0; N_ENGINES];
                engines[0] = p.entry_throttle;
                engines[1] = p.entry_throttle;
                engines[2] = p.entry_throttle;
                cmd.engines = engines;
                // Fixed delta-v cutoff (the recorded 367 m/s): keeping the
                // entry burn repeatable makes its IIP pullback repeatable,
                // which the boostback overshoot bias absorbs (WHITEPAPER 11.4).
                if self.entry_v0 - speed >= p.entry_dv_mps {
                    let (along, cross) = self.iip_miss_components(s, up_here);
                    eprintln!("[fsw] entry cutoff: iip miss along={along:.0} cross={cross:.0} m");
                    self.cutoff_with_purge(&mut cmd, t);
                    self.set_phase(Phase::AeroDescent, t);
                }
            }
            Phase::AeroDescent => {
                // Grid fins own attitude in thick air (holding an AoA against
                // the static margin continuously would drain the cold-gas
                // budget); the RCS backstops in thin air.
                let qbar = 0.5 * density(alt) * speed * speed;
                cmd.rcs_enable = if qbar > 2_000.0 { 0.0 } else { 1.0 };
                // Retrograde, tilted to steer the drag vector toward LZ-1.
                let steer = self.descent_steer(s, up_here);
                let retro = normalize(add(scale(normalize(self.nav.vel), -1.0), steer));
                let desired = quat_between([1.0, 0.0, 0.0], retro);
                cmd.attitude = self.slew_attitude(desired, 0.01);
                cmd.fins = self.fin_attitude_pd(retro, s, false);
                // Hoverslam ignition (WHITEPAPER 11.5): ignite when the
                // descent rate reaches the three-engine opening profile,
                // charging ~2.5 s of spool-up distance against the altitude.
                let vdown = -dot(self.nav.vel, up_here);
                let a_land = 0.70 * self.landing_accel_net(s, 3.0);
                let h_eff = (alt - 2.5 * vdown.max(0.0) - 20.0).max(1.0);
                let v_profile = (2.0 * a_land * h_eff).sqrt();
                if t - self.phase_t0 > 5.0
                    && ((t - self.phase_t0) as u64).is_multiple_of(10)
                    && dt_edge(t)
                {
                    let (along, cross) = self.iip_miss_components(s, up_here);
                    eprintln!(
                        "[fsw] t={t:6.1} descent: iip miss along={along:.0} cross={cross:.0} m, alt {alt:.0}"
                    );
                }
                if alt <= p.landing_arm_alt_m && vdown * p.landing_accel_margin >= v_profile {
                    let (along, cross) = self.iip_miss_components(s, up_here);
                    eprintln!(
                        "[fsw] landing ignition: iip miss along={along:.0} cross={cross:.0} m"
                    );
                    self.landing_escalated = true; // 3-engine start (hot RTLS)
                    self.set_phase(Phase::LandingBurn, t);
                }
            }
            Phase::LandingBurn => {
                cmd.valves[V_MAIN_LOX] = 1.0;
                cmd.valves[V_MAIN_RP1] = 1.0;
                cmd.valves[V_TEATEB] = 1.0;
                cmd.tvc_enable = 1.0;
                cmd.rcs_enable = 1.0;
                // 3→1 engine profile (TEA-TEB): open on three, hand over to
                // the center engine once a single-engine profile can finish.
                let mass = 25_600.0 + s.lox_kg + s.rp1_kg;
                let vdown = -dot(self.nav.vel, up_here);
                let h = (alt - 2.0).max(0.5);
                let t_single_min = THROTTLE_MIN * T_VAC_PER_ENGINE - 101_325.0 * A_E_M2;
                let a_floor = (t_single_min / mass - 9.81).max(0.5);
                let a_single = self.landing_accel_net(s, 1.0);
                let a_mid = 0.5 * (a_floor + a_single);
                if self.landing_escalated
                    && !self.landing_deescalated
                    && vdown <= (2.0 * a_mid * h).sqrt() + 1.0
                {
                    self.landing_deescalated = true;
                }
                let three = self.landing_escalated && !self.landing_deescalated;
                let (n, a_land) = if three {
                    (3.0, 0.70 * self.landing_accel_net(s, 3.0))
                } else {
                    (1.0, a_mid)
                };
                // Continuous hoverslam vertical (WHITEAPER 11.5): T_min/W > 1
                // so we never coast mid-burn — the rate loop keeps vdown on the
                // suicide curve. ZEM/ZEV only shapes the thrust *direction*.
                let v_des = (2.0 * a_land * h).sqrt() + ZEM_V_TD_MPS;
                // Slightly higher rate gain in the last 200 m to keep impact ≤ 2 m/s.
                let kv = if alt < 200.0 { 4.0 } else { 3.2 };
                let a_up = (9.81 + kv * (vdown - v_des)).max(0.0);

                let (t_go, t_raw) = Self::t_go_hoverslam(h, vdown.max(1.0));
                let miss_h = {
                    let d = sub(self.lz1_pos, self.nav.pos);
                    norm(sub(d, scale(up_here, dot(d, up_here))))
                };
                // Commit-to-vertical once near the pad *and* the divert is
                // mostly closed — don't freeze lateral with a large miss left.
                if !self.landing_vertical_commit {
                    let time_gate = t_raw > 0.0 && t_raw < ZEM_COMMIT_TGO_S && alt < 200.0;
                    let alt_gate = alt < ZEM_COMMIT_ALT_M;
                    if (alt_gate || time_gate) && (miss_h < 25.0 || alt < 25.0) {
                        self.landing_vertical_commit = true;
                    }
                }
                let a_zem = self.zem_zev_accel(up_here, t_go, self.landing_vertical_commit);
                let a_lat = if self.landing_vertical_commit {
                    [0.0; 3]
                } else {
                    let a_up_zem = dot(a_zem, up_here);
                    sub(a_zem, scale(up_here, a_up_zem))
                };
                let lat_mag = norm(a_lat);
                let max_lat = (a_up.max(9.81)) * ZEM_TILT_CAP_RAD.tan();
                let a_lat = if lat_mag > max_lat && lat_mag > 1e-6 {
                    scale(normalize(a_lat), max_lat)
                } else {
                    a_lat
                };
                let a_cmd = add(scale(up_here, a_up.max(9.81)), a_lat);
                let dir = normalize(a_cmd);
                let desired = quat_between([1.0, 0.0, 0.0], dir);
                cmd.attitude = self.slew_attitude(desired, 0.01);
                let body_x = quat_rotate(cmd.attitude, [1.0, 0.0, 0.0]);
                cmd.fins = self.fin_attitude_pd(body_x, s, true);

                let cos_tilt = dot(body_x, up_here).max(0.6);
                let u = ((mass * a_up / cos_tilt / n + 101_325.0 * A_E_M2) / T_VAC_PER_ENGINE)
                    .clamp(THROTTLE_MIN, 1.0);
                let mut engines = [0.0; N_ENGINES];
                // If min-throttle has lofted us (T_min/W > 1), cut until we
                // are descending again — otherwise we climb away from the pad.
                let lofting = alt < 100.0 && vdown < -0.5;
                if !lofting {
                    engines[0] = u;
                    if three {
                        engines[1] = u;
                        engines[2] = u;
                    }
                }
                cmd.engines = engines;
                if s.landed > 0.5 || (alt < 2.0 && speed < 1.5) {
                    self.cutoff_with_purge(&mut cmd, t);
                    self.set_phase(Phase::Touchdown, t);
                }
            }
            Phase::Touchdown => {
                cmd.engines = [0.0; N_ENGINES];
                cmd.attitude = quat_between([1.0, 0.0, 0.0], up_here);
            }
        }
        cmd.phase = self.phase as i32 as f64;
        cmd
    }

    /// Drag-aware ballistic impact point via forward integration of the same
    /// rotating-frame EOM plus an engines-first drag model (WHITEPAPER 11.3).
    /// The vacuum IIP would overshoot by tens of km — descent drag steepens
    /// the fall — so the predictor carries the FSW's own aero estimate.
    fn impact_point(&self, s: &SensorPacket) -> V3 {
        let mass = 25_600.0 + s.lox_kg + s.rp1_kg;
        let cd_s = if s.params.fsw_cd_s_m2 > 1.0 {
            s.params.fsw_cd_s_m2
        } else {
            22.0
        };
        let mut r = self.nav.pos;
        let mut v = self.nav.vel;
        let dt = 0.5;
        for _ in 0..2400 {
            let alt = ecef_to_geodetic(r).2;
            if alt <= 0.0 {
                return r;
            }
            let speed = norm(v);
            let a_drag = if speed > 1.0 {
                scale(
                    normalize(v),
                    -0.5 * density(alt) * speed * speed * cd_s / mass,
                )
            } else {
                [0.0; 3]
            };
            let a = add(add(gravity(r), frame_accel(r, v)), a_drag);
            v = add(v, scale(a, dt));
            r = add(r, scale(v, dt));
        }
        r
    }

    /// Boostback burn solution: aim the predicted impact point at a target
    /// `boostback_overshoot_m` beyond LZ-1 along the approach course (the
    /// entry burn pulls the IIP back by about that much). Returns the burn
    /// direction and the horizontal IIP-to-target distance.
    fn boostback_solution(&self, s: &SensorPacket, up: V3, p: &GuidanceParams) -> (V3, f64) {
        let iip = self.impact_point(s);
        // Approach course: from LZ-1 back toward the booster's ground point,
        // extended beyond the pad — i.e. the direction the vehicle will be
        // flying when it arrives. Overshoot lies past LZ-1 along that course.
        let here = sub(
            self.nav.pos,
            scale(up, dot(sub(self.nav.pos, self.lz1_pos), up)),
        );
        let course = normalize(sub(self.lz1_pos, here)); // horizontal-ish
        let course_h = normalize(sub(course, scale(up, dot(course, up))));
        let target = add(self.lz1_pos, scale(course_h, p.boostback_overshoot_m));
        let miss = sub(target, iip);
        let miss_h = sub(miss, scale(up, dot(miss, up)));
        let mag = norm(miss_h);
        let dir = if mag > 1e-6 {
            normalize(sub(normalize(miss_h), scale(up, 0.10)))
        } else {
            scale(normalize(self.nav.vel), -1.0)
        };
        (dir, mag)
    }

    /// Signed IIP miss components: along the approach course (positive =
    /// undershoot, impact before LZ-1) and cross-course.
    fn iip_miss_components(&self, s: &SensorPacket, up: V3) -> (f64, f64) {
        let iip = self.impact_point(s);
        let v_h = sub(self.nav.vel, scale(up, dot(self.nav.vel, up)));
        let course = normalize(v_h);
        let miss = sub(self.lz1_pos, iip);
        let miss_h = sub(miss, scale(up, dot(miss, up)));
        let along = dot(miss_h, course);
        let cross_vec = sub(miss_h, scale(course, along));
        let cross_sign = if dot(cross(up, course), cross_vec) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        (along, cross_sign * norm(cross_vec))
    }

    /// Drag-vector steering during aero descent (WHITEPAPER 11.4).
    /// Cross-track PD plus along-track AoA modulation. Engines-first CA > CN
    /// so any AoA lowers total deceleration (stretches the IIP). Tilt cap is
    /// scheduled down at high q̄ so Max-Q passes near zero AoA.
    fn descent_steer(&self, s: &SensorPacket, up: V3) -> V3 {
        let iip = self.impact_point(s);
        let miss = sub(self.lz1_pos, iip);
        let miss_h = sub(miss, scale(up, dot(miss, up)));
        let v_h = sub(self.nav.vel, scale(up, dot(self.nav.vel, up)));
        let course = normalize(v_h);
        let cross_err = sub(miss_h, scale(course, dot(miss_h, course)));
        let cross_vel = sub(v_h, scale(course, dot(v_h, course)));
        let pd = sub(
            scale(cross_err, 1.0 / 2_500.0),
            scale(cross_vel, 1.0 / 50.0),
        );
        let mag = norm(pd);
        let speed = self.nav.speed();
        let qbar = 0.5 * density(self.nav.altitude()) * speed * speed;
        // Full steer_tilt_cap below ~25 kPa; fade toward ~0.03 rad by 60 kPa+.
        let base_cap = s.params.steer_tilt_cap;
        // Full authority below ~30 kPa; fade toward 0.04 rad by 70 kPa+.
        let tilt_cap = (base_cap * (30_000.0 / qbar.max(5_000.0))).clamp(0.04, base_cap);
        let (along, _) = self.iip_miss_components(s, up);
        let cross_quiet = norm(cross_err) < 80.0 && norm(cross_vel) < 2.0;
        let tilt = if cross_quiet { 0.0 } else { mag.min(tilt_cap) };
        // Positive tilt along the PD direction deflects the drag vector toward
        // the target (fins have the correct plant sign on pitch).
        let cross_dir = if tilt > 1e-6 {
            scale(normalize(pd), tilt)
        } else {
            [0.0; 3]
        };
        // Along-track drag modulation: only add AoA when undershooting
        // (stretch range). Overshoot → 0 (max CA drag).
        let aoa_along = (along / 8_000.0).clamp(0.0, 0.10);
        let retro = scale(normalize(self.nav.vel), -1.0);
        let lift_dir = normalize(sub(course, scale(retro, dot(course, retro))));
        let along_tilt = scale(lift_dir, aoa_along);
        add(cross_dir, along_tilt)
    }

    /// Torque-based fin PD: desired angular acceleration → deflection via
    /// q̄·effectiveness, so closed-loop bandwidth stays near `fin_wn` across
    /// the descent q̄ range. `rate_only` drops the attitude term (LandingBurn
    /// under TVC — fins damp residual rates only).
    fn fin_attitude_pd(&self, desired_dir: V3, s: &SensorPacket, rate_only: bool) -> [f64; 3] {
        let x_body = quat_rotate(self.nav.att, [1.0, 0.0, 0.0]);
        let err_world = cross(x_body, desired_dir);
        let err_body = quat_rotate_inv(self.nav.att, err_world);
        let wn = s.params.fin_wn;
        let zeta = 0.85;
        let err = if rate_only {
            [0.0, 0.0, 0.0]
        } else {
            err_body
        };
        let alpha = [
            wn * wn * err[0] - 2.0 * zeta * wn * s.imu_gyro[0],
            wn * wn * err[1] - 2.0 * zeta * wn * s.imu_gyro[1],
            wn * wn * err[2] - 2.0 * zeta * wn * s.imu_gyro[2],
        ];
        let speed = self.nav.speed().max(1.0);
        let qbar = 0.5 * density(self.nav.altitude()) * speed * speed;
        // Floor keeps authority in thin air; pair of fins × lever.
        let q_eff = qbar.max(2_000.0);
        // Plant mapping (aero.fin_mix / fin_wrench): +pitch → −My, +yaw → +Mz.
        let k_defl = 2.0 * q_eff * FIN_S_M2 * FIN_CN_DELTA * FIN_LEVER_M;
        let pitch = (-FIN_I_TRANS * alpha[1] / k_defl).clamp(-0.35, 0.35);
        let yaw = (FIN_I_TRANS * alpha[2] / k_defl).clamp(-0.35, 0.35);
        let roll = (FIN_I_TRANS * alpha[0] / k_defl).clamp(-0.35, 0.35);
        [pitch, yaw, roll]
    }

    fn landing_accel_net(&self, s: &SensorPacket, n_engines: f64) -> f64 {
        // Near-full throttle at the estimated landing mass (propellant
        // knowledge comes from the FSW's own flow accounting; the bridge
        // forwards it as config).
        let mass = 25_600.0 + s.lox_kg + s.rp1_kg;
        let thrust = n_engines * 0.85 * (T_VAC_PER_ENGINE - 101_325.0 * A_E_M2);
        (thrust / mass - 9.81).max(1.0)
    }

    /// Hoverslam-consistent time-to-go: `(t_go_clamped, t_raw)`.
    fn t_go_hoverslam(h: f64, vdown: f64) -> (f64, f64) {
        let h = h.max(0.5);
        let vdown = vdown.max(0.1);
        let a_req = (vdown * vdown - ZEM_V_TD_MPS * ZEM_V_TD_MPS).max(0.0) / (2.0 * h);
        let a_use = a_req.clamp(0.5, ZEM_A_LAND_TGO);
        let t_raw = (vdown - ZEM_V_TD_MPS) / a_use;
        (t_raw.clamp(0.5, 80.0), t_raw)
    }

    /// ZEM/ZEV thrust-acceleration command (ECEF). Target is LZ-1 with a
    /// 150 m / −25 m/s waypoint until crossed, then pad / −1.5 m/s.
    fn zem_zev_accel(&self, up: V3, t_go: f64, commit: bool) -> V3 {
        if commit {
            let vdown = -dot(self.nav.vel, up);
            let a_up = 9.81 + 3.0 * (vdown - ZEM_V_TD_MPS);
            return scale(up, a_up);
        }
        // Wind-bias the aim point upwind by the expected terminal drift.
        let wind_h = sub(self.nav.wind_est, scale(up, dot(self.nav.wind_est, up)));
        let aim = sub(self.lz1_pos, scale(wind_h, t_go.min(25.0)));
        let r = sub(self.nav.pos, aim);
        let v = self.nav.vel;
        let alt = self.nav.altitude();
        let g_vec = scale(up, -9.81);
        let (r_tgt, v_tgt) = if alt > ZEM_WAYPOINT_ALT_M {
            (scale(up, ZEM_WAYPOINT_ALT_M), scale(up, -ZEM_WAYPOINT_VDOWN_MPS))
        } else {
            ([0.0; 3], scale(up, -ZEM_V_TD_MPS))
        };
        let t2 = t_go * t_go;
        let zem = sub(
            r_tgt,
            add(add(r, scale(v, t_go)), scale(g_vec, 0.5 * t2)),
        );
        let zev = sub(v_tgt, add(v, scale(g_vec, t_go)));
        // a_cmd = 6 ZEM/t² − 2 ZEV/t − g  (thrust accel; plant adds gravity)
        sub(
            add(scale(zem, 6.0 / t2), scale(zev, -2.0 / t_go)),
            g_vec,
        )
    }
}

/// True on ~one guidance tick per second (for rate-limited diagnostics).
fn dt_edge(t: f64) -> bool {
    (t - t.floor()) < 0.011
}

fn port_from_env(name: &str, default: u16) -> u16 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> std::io::Result<()> {
    let state_port = port_from_env("ELODIN_MC_PORT_STATE", 9114);
    let command_port = port_from_env("ELODIN_MC_PORT_COMMAND", 9115);
    let sock = UdpSocket::bind(("127.0.0.1", state_port))?;
    eprintln!("[fsw] listening on 127.0.0.1:{state_port}, replying to :{command_port}");

    let mut fsw = Fsw::new();
    let mut buf = [0u8; STATE_FLOATS * 8];
    loop {
        let (n, _src) = sock.recv_from(&mut buf)?;
        if n < STATE_FLOATS * 8 {
            continue;
        }
        let mut vals = [0.0f64; STATE_FLOATS];
        for (i, v) in vals.iter_mut().enumerate() {
            *v = f64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
        }
        let packet = SensorPacket::parse(&vals);
        let cmd = fsw.step(&packet).pack();
        let mut out = [0u8; CMD_FLOATS * 8];
        for (i, v) in cmd.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        sock.send_to(&out, ("127.0.0.1", command_port))?;
    }
}
