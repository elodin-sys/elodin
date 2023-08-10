use crate::{
    effector::{Effector, ErasedStateEffector, StateEffect},
    Force, FromState, Mass, Pos, Time, Torque,
};
use nalgebra::{Quaternion, Vector3};
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul},
};

#[derive(Default, Debug, Clone, Copy)]
pub struct SixDof {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub mass: f64,
    pub ang: Quaternion<f64>,
    pub ang_vel: Vector3<f64>,
    pub time: f64,
}

impl SixDof {
    pub fn sim(self) -> Sim<SixDof> {
        Sim::new(self)
    }
    pub fn pos(mut self, pos: Vector3<f64>) -> Self {
        self.pos = pos;
        self
    }
    pub fn vel(mut self, vel: Vector3<f64>) -> Self {
        self.vel = vel;
        self
    }
    pub fn mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self
    }
    pub fn ang(mut self, ang: Quaternion<f64>) -> Self {
        self.ang = ang;
        self
    }
    pub fn ang_vel(mut self, ang_vel: Vector3<f64>) -> Self {
        self.ang_vel = ang_vel;
        self
    }
    pub fn time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }
}

impl Add for SixDof {
    type Output = SixDof;

    fn add(self, rhs: Self) -> Self::Output {
        SixDof {
            pos: self.pos + rhs.pos,
            vel: self.vel + rhs.vel,
            mass: self.mass + rhs.mass,
            ang: self.ang + rhs.ang,
            ang_vel: self.ang_vel + rhs.ang_vel,
            time: self.time + rhs.time,
        }
    }
}

impl Mul<SixDof> for f64 {
    type Output = SixDof;

    fn mul(self, rhs: SixDof) -> Self::Output {
        SixDof {
            pos: rhs.pos * self,
            vel: rhs.vel * self,
            mass: rhs.mass * self,
            ang: rhs.ang * self,
            ang_vel: rhs.ang_vel * self,
            time: rhs.time * self,
        }
    }
}

impl FromState<SixDof> for Mass {
    fn from_state(_time: Time, state: &SixDof) -> Self {
        Mass(state.mass)
    }
}

impl FromState<SixDof> for Pos {
    fn from_state(_time: Time, state: &SixDof) -> Self {
        Pos(state.pos)
    }
}

impl FromState<SixDof> for Time {
    fn from_state(time: Time, _state: &SixDof) -> Self {
        time
    }
}

impl StateEffect<SixDof> for Force {
    fn apply(&self, _time: Time, init_state: &SixDof, inc_state: &mut SixDof) {
        let accl = self.0 / init_state.mass;
        inc_state.vel += accl;
    }
}

impl StateEffect<SixDof> for Torque {
    fn apply(&self, _time: Time, init_state: &SixDof, inc_state: &mut SixDof) {
        let ang_accl = self.0 / init_state.mass; // TODO: use moment of inertia
        inc_state.ang_vel += ang_accl;
    }
}

impl State for SixDof {
    fn step(&self, inc_state: &mut SixDof) {
        inc_state.pos += self.vel;
        inc_state.ang =
            0.5 * Quaternion::new(0., self.ang_vel.x, self.ang_vel.y, self.ang_vel.z) * self.ang;
        // NOTE: This relies on the small angle approx a bunch, I think this is safe for small dt, but for very high angular vel it will fail
        inc_state.time += 1.0;
    }
}

impl AddAssign for SixDof {
    fn add_assign(&mut self, rhs: Self) {
        self.pos += rhs.pos;
        self.vel += rhs.vel;
        self.mass += rhs.mass;
        self.ang += rhs.ang;
        self.ang_vel += rhs.ang_vel;
        self.time += rhs.time;
    }
}

pub trait TimeState {
    fn time(&self) -> Time;
}

impl TimeState for SixDof {
    fn time(&self) -> Time {
        Time(self.time)
    }
}

pub struct Sim<S> {
    pub state: S,
    effectors: Vec<Box<dyn StateEffect<S>>>,
}

impl<S: Debug> Debug for Sim<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sim").field("state", &self.state).finish()
    }
}

impl<S> Sim<S>
where
    S: Default
        + Clone
        + Copy
        + Debug
        + Add<Output = S>
        + AddAssign<S>
        + State
        + TimeState
        + 'static,
    f64: Mul<S, Output = S>,
{
    pub fn new(state: S) -> Self {
        Self {
            state,
            effectors: vec![],
        }
    }

    pub fn effector<T, E, EF>(mut self, effector: E) -> Self
    where
        T: 'static,
        E: 'static,
        E: Effector<T, S, Effect = EF> + 'static,
        EF: StateEffect<S> + 'static,
    {
        self.effectors.push(ErasedStateEffector::boxed(effector));
        self
    }

    pub fn tick(&mut self, dt: f64) {
        let delta = rk4_step(
            |init_state| {
                let mut state = self.effectors.iter().fold(S::default(), |mut s, e| {
                    e.apply(init_state.time(), &init_state, &mut s);
                    s
                });
                init_state.step(&mut state);
                state
            },
            self.state,
            dt,
        );
        self.state += delta;
    }
}

pub fn rk4_step<F, S, A>(f: F, init_state: S, dt: f64) -> A
where
    F: Fn(S) -> A,
    S: Add<A, Output = S> + Copy + Debug,
    A: Add<A, Output = A> + Copy + Debug,
    f64: Mul<A, Output = A>,
{
    let k1 = f(init_state);
    let half_dt: f64 = dt / 2.0;
    let k2 = f(init_state + half_dt * k1);
    let k3 = f(init_state + half_dt * k2);
    let k4 = f(init_state + dt * k3);
    (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
}

pub trait State {
    fn step(&self, inc_state: &mut Self); // TODO: this feels kinda hacky or maybe just a bad name
}

#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;
    use nalgebra::{UnitQuaternion, UnitVector3};

    use super::*;

    #[test]
    fn impulse_torque() {
        let mut sim = SixDof::default()
            .mass(1.0)
            .ang(*UnitQuaternion::from_axis_angle(
                &UnitVector3::new_unchecked(Vector3::new(0.0, 1.0, 0.0)),
                0.0,
            ))
            .sim()
            .effector(|Time(t)| {
                if t <= 1.0 {
                    Torque(Vector3::new(1.0, 0.0, 0.0))
                } else {
                    Torque(Vector3::zeros())
                }
            });
        const DT: f64 = 0.01;
        for _ in 0..100 {
            sim.tick(DT);
        }
        assert_relative_eq!(
            sim.state.ang,
            *UnitQuaternion::from_axis_angle(
                &UnitVector3::new_unchecked(Vector3::new(1.0, 0.0, 0.0)),
                0.5,
            ),
            epsilon = 1e-9,
        )
    }
}
