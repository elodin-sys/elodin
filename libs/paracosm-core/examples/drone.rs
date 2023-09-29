use bevy_ecs::system::Resource;
use nalgebra::{vector, UnitQuaternion, Vector3, Vector4};
use paracosm::{
    forces::earth_gravity,
    xpbd::{
        builder::{Assets, EntityBuilder, XpbdBuilder},
        components::Effect,
        editor::editor,
        editor::Editable,
        runner::IntoSimRunner,
    },
    Att, Force, Pos, SharedNum, Time, Vel,
};
use paracosm_macros::Editable;

const G: f64 = 9.81;

fn main() {
    editor(sim.substep_count(8))
}

fn sim(
    mut builder: XpbdBuilder<'_>,
    mut assets: Assets,
    motor_a_rpm: MotorARPM,
    motor_b_rpm: MotorBRPM,
    motor_c_rpm: MotorCRPM,
    motor_d_rpm: MotorDRPM,
    alt_set_point: Alt,
) {
    let default_rpm = (9.81 / 4.0 / MOTOR_B).sqrt();
    let motor_a_pos = vector![-0.25, 0.0, 0.25];
    let motor_b_pos = vector![0.25, 0.0, 0.25];
    let motor_c_pos = vector![-0.25, 0.0, -0.25];
    let motor_d_pos = vector![0.25, 0.0, -0.25];
    *alt_set_point.0.load() = 2.5;
    let (tx, rx) = flume::unbounded();
    {
        *motor_a_rpm.0.load() = -default_rpm;
        *motor_b_rpm.0.load() = default_rpm;
        *motor_c_rpm.0.load() = -default_rpm;
        *motor_d_rpm.0.load() = default_rpm;
    }
    FSWActor {
        pid_controller: PIDController {
            state: Vector3::zeros(),
            integrated_state: Vector3::zeros(),
            integrated_alt: 0.0,
            alt: 0.0,
            last_time: 0.0,
            alt_set_point,
        },
        rx,
        motor_a: motor_a_rpm.clone(),
        motor_b: motor_b_rpm.clone(),
        motor_c: motor_c_rpm.clone(),
        motor_d: motor_d_rpm.clone(),
    }
    .run();
    builder.entity(
        EntityBuilder::default()
            .mass(1.0)
            .model(assets.load("/Users/sphw/drone.glb#Scene0"))
            .effector(earth_gravity)
            .effector(move |Vel(v)| Force(-v * 0.5 * v.norm()))
            .effector(move |Att(a)| motor(*motor_a_rpm.0.load(), a, motor_a_pos))
            .effector(move |Att(a)| motor(*motor_b_rpm.0.load(), a, motor_b_pos))
            .effector(move |Att(a)| motor(*motor_c_rpm.0.load(), a, motor_c_pos))
            .effector(move |Att(a)| motor(*motor_d_rpm.0.load(), a, motor_d_pos))
            .sensor(move |Att(a), Pos(p), Vel(v), Time(time)| {
                let (yaw, pitch, roll) = a.euler_angles();
                //let euler_angles = Vector3::new(roll, pitch, yaw);
                let euler_angles = Vector3::new(yaw, pitch, roll);
                tx.send(SensorInputs {
                    euler_angles,
                    alt: p.y,
                    alt_dot: v.y,
                    time,
                })
                .unwrap();
            }),
    );

    builder.entity(
        EntityBuilder::default()
            .model(assets.load("/Users/sphw/tower.glb#Scene0"))
            .mass(1.0)
            .pos(vector![10.0, 39.0, 0.0]),
    );

    builder.entity(
        EntityBuilder::default()
            .model(assets.load("/Users/sphw/rotor.glb#Scene0"))
            .mass(1.0)
            .ang_vel(vector![0.0, 0.0, 2.0])
            .pos(vector![10.0, 84.0, 18.4 / 2.0 + 2.0]),
    );
    builder.entity(
        EntityBuilder::default()
            .model(assets.load("/Users/sphw/turbine.glb#Scene0"))
            .mass(1.0)
            .pos(vector![10.0, 81.0, 0.0]),
    );
}

const MOTOR_B: f64 = 2e-6 / 1000.0 * G;

fn motor(rpm: f64, att: UnitQuaternion<f64>, loc: Vector3<f64>) -> Effect {
    let thrust = rpm.powi(2) * MOTOR_B;
    let k = rpm.signum();
    let mut body_effect = force_at_point(loc, vector![0.0, thrust, 0.0], att);
    body_effect.torque.0 += att * vector![0.0, k * thrust, 0.0];
    body_effect
}

fn force_at_point(r: Vector3<f64>, force: Vector3<f64>, att: UnitQuaternion<f64>) -> Effect {
    Effect {
        torque: paracosm::Torque(att * r.cross(&force)),
        force: Force(att * force),
    }
}

#[derive(Editable, Resource, Clone, Debug, Default)]
#[editable(slider, range_min = "-20800", range_max = 20800.0, name = "motor a")]
pub struct MotorARPM(pub SharedNum<f64>);

#[derive(Editable, Resource, Clone, Debug, Default)]
#[editable(slider, range_min = "-20800", range_max = 20800.0, name = "motor b")]
pub struct MotorBRPM(pub SharedNum<f64>);

#[derive(Editable, Resource, Clone, Debug, Default)]
#[editable(slider, range_min = "-20080", range_max = 20800.0, name = "motor c")]
pub struct MotorCRPM(pub SharedNum<f64>);

#[derive(Editable, Resource, Clone, Debug, Default)]
#[editable(slider, range_min = "-20080", range_max = 20800.0, name = "motor d")]
pub struct MotorDRPM(pub SharedNum<f64>);

#[derive(Editable, Resource, Clone, Debug, Default)]
#[editable(slider, range_min = "0", range_max = 90.0, name = "alt")]
pub struct Alt(pub SharedNum<f64>);

struct FSWActor {
    pid_controller: PIDController,
    rx: flume::Receiver<SensorInputs>,
    motor_a: MotorARPM,
    motor_b: MotorBRPM,
    motor_c: MotorCRPM,
    motor_d: MotorDRPM,
}

struct SensorInputs {
    euler_angles: Vector3<f64>,
    alt: f64,
    alt_dot: f64,
    time: f64,
}

impl FSWActor {
    fn run(mut self) {
        std::thread::spawn(move || {
            while let Ok(input) = self.rx.recv() {
                let motors = self.pid_controller.step(input);
                let motors: Vec<f64> = motors
                    .as_slice()
                    .into_iter()
                    .copied()
                    .map(f64::sqrt)
                    .collect();
                if !motors.iter().copied().all(f64::is_finite) {
                    continue;
                }

                *self.motor_a.0.load() = -1.0 * motors[0];
                *self.motor_b.0.load() = motors[1];
                *self.motor_c.0.load() = -1.0 * motors[2];
                *self.motor_d.0.load() = motors[3];
            }
        });
    }
}

struct PIDController {
    state: Vector3<f64>, // roll (ɸ), pitch (Θ), yaw (ψ)
    integrated_state: Vector3<f64>,

    integrated_alt: f64,
    alt: f64,
    last_time: f64,
    alt_set_point: Alt,
}

impl PIDController {
    const P: f64 = -0.1;
    const I: f64 = 0.0;
    const D: f64 = -0.01;

    const P_ALT: f64 = 1.0;
    const I_ALT: f64 = 0.0;
    const D_ALT: f64 = -1.0;

    fn step(&mut self, input: SensorInputs) -> Vector4<f64> {
        // TODO: prevent wind up
        let dt = input.time - self.last_time;
        if dt == 0.0 {
            return Vector4::zeros();
        }

        let alt_err = *self.alt_set_point.0.load() - input.alt;

        let d_alt = input.alt_dot;

        let default_rpm = 9.81 / 4.0 / MOTOR_B;

        let alt_pid =
            Self::P_ALT * alt_err + Self::I_ALT * self.integrated_alt + Self::D_ALT * d_alt;
        let alt_pid = 1.0 + alt_pid;

        let d_state = (self.state - input.euler_angles) / dt;

        let p = Self::P * input.euler_angles;
        let i = Self::I * self.integrated_state;
        let d = Self::D * d_state;
        let pid = p + i + d;

        let motor_a = (alt_pid - pid[1] - pid[2]) * default_rpm;
        let motor_b = (alt_pid - pid[0] + pid[2]) * default_rpm;
        let motor_c = (alt_pid + pid[1] - pid[2]) * default_rpm;
        let motor_d = (alt_pid + pid[0] + pid[2]) * default_rpm;

        self.state = input.euler_angles;
        self.last_time = input.time;
        self.alt = alt_err;
        self.integrated_alt += dt * alt_err;
        self.integrated_state += dt * input.euler_angles;
        vector![motor_a, motor_b, motor_c, motor_d]
    }
}
