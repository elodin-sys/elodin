use std::ops::{Add, AddAssign, Mul};

use nalgebra::{Quaternion, Vector3};

use crate::{Force, FromState, Mass, Pos, Sim, State, StateEffect, Time};

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
    fn from_state(state: &SixDof) -> Self {
        Mass(state.mass)
    }
}

impl FromState<SixDof> for Pos {
    fn from_state(state: &SixDof) -> Self {
        Pos(state.pos)
    }
}

impl FromState<SixDof> for Time {
    fn from_state(state: &SixDof) -> Self {
        Time(state.time)
    }
}

impl StateEffect<SixDof> for Force {
    fn apply(&self, init_state: &SixDof, inc_state: &mut SixDof) {
        let accl = self.0 / init_state.mass;
        inc_state.vel += accl;
    }
}

impl State for SixDof {
    fn step(&self, inc_state: &mut SixDof) {
        inc_state.pos += self.vel;
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
