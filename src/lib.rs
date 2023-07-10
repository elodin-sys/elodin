use nalgebra::{Quaternion, Vector3};
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Add, AddAssign, Mul},
};

pub trait Effector<T, S> {
    type Effect;

    fn effect(&self, state: &S) -> Self::Effect;
}

pub trait StateEffect<S> {
    fn apply(&self, init_state: &S, inc_state: &mut S);
}

struct ErasedEffector<T, S, ER> {
    effector: ER,
    _phantom: PhantomData<(T, S)>,
}

impl<T, ER, E, S> ErasedEffector<T, S, ER>
where
    ER: Effector<T, S, Effect = E> + 'static,
    E: StateEffect<S> + 'static,
    S: 'static,
    T: 'static,
{
    fn new(effector: ER) -> Box<dyn StateEffect<S>> {
        Box::new(ErasedEffector {
            effector,
            _phantom: PhantomData,
        })
    }
}

impl<T, ER: Effector<T, S, Effect = E>, E: StateEffect<S>, S> StateEffect<S>
    for ErasedEffector<T, S, ER>
{
    fn apply(&self, init_state: &S, inc_state: &mut S) {
        let effect = self.effector.effect(init_state);
        effect.apply(init_state, inc_state)
    }
}

macro_rules! impl_effector {
    ($($ty:tt),+) => {
        #[allow(non_snake_case)]
        impl<F, $($ty,)* E, S> Effector<($($ty, )*), S> for F
        where
            F: Fn($($ty, )*) -> E,
            $($ty: FromState<S>, )*
        {
            type Effect = E;

            fn effect(&self, state: &S) -> Self::Effect {
                $(
                    let $ty = $ty::from_state(&state);
                )*
                (self)($($ty,)*)
            }
        }
    };
}

impl_effector!(T1);
impl_effector!(T1, T2);
impl_effector!(T1, T2, T3);
impl_effector!(T1, T2, T3, T4);
impl_effector!(T1, T2, T3, T4, T5);
impl_effector!(T1, T2, T3, T4, T5, T6);
impl_effector!(T1, T2, T3, T4, T5, T6, T7);
impl_effector!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_effector!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_effector!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_effector!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_effector!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

pub trait FromState<S> {
    fn from_state(state: &S) -> Self;
}

pub struct Sim<S> {
    pub state: S,
    effectors: Vec<Box<dyn StateEffect<S>>>,
}

impl<S> Sim<S>
where
    S: Default + Clone + Copy + Debug + Add<Output = S> + AddAssign<S> + State + 'static,
    f64: Mul<S, Output = S>,
{
    pub fn new(state: S) -> Self {
        Self {
            state,
            effectors: vec![],
        }
    }

    pub fn add_effector<T, E, EF>(&mut self, effector: E)
    where
        T: 'static,
        E: 'static,
        E: Effector<T, S, Effect = EF> + 'static,
        EF: StateEffect<S> + 'static,
    {
        self.effectors.push(ErasedEffector::new(effector));
    }

    pub fn tick(&mut self, dt: f64) {
        let delta = rk4_step(
            |init_state| {
                let mut state = self.effectors.iter().fold(S::default(), |mut s, e| {
                    e.apply(&init_state, &mut s);
                    s
                });
                init_state.step(&mut state, dt);
                state
            },
            self.state.clone(),
            dt,
        );
        println!("{:?}", delta);
        self.state += delta;
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct SixDof {
    pub pos: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub mass: f64,
    pub ang: Quaternion<f64>,
    pub ang_vel: Vector3<f64>,
    pub time: f64,
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

pub fn gravity(
    body_mass: f64,
    body_pos: Vector3<f64>,
) -> impl Effector<(Mass, Pos), SixDof, Effect = Force> {
    move |Mass(m), Pos(pos)| {
        const G: f64 = 6.649e-11;
        let r = body_pos - pos;
        let mu = G * body_mass;
        let f = Force(r * (mu * m / r.norm().powi(3)));
        f
    }
}

#[derive(Debug)]
pub struct Force(pub Vector3<f64>);
pub struct Mass(pub f64);
pub struct Pos(Vector3<f64>);
pub struct Time(pub f64);

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

pub trait State {
    fn step(&self, inc_state: &mut Self, dt: f64); // TODO: this feels kinda hacky or maybe just a bad name
}

impl State for SixDof {
    fn step(&self, inc_state: &mut SixDof, dt: f64) {
        inc_state.pos += self.vel;
        inc_state.time += dt;
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use plotters::prelude::*;

    #[test]
    fn test_const_rk4() {
        let mut int = 0.0;
        for _ in 0..10 {
            int += rk4_step(|_| 1., 0., 0.1);
        }
        assert_relative_eq!(int, 1.0);
    }

    #[test]
    fn test_perfect_simple_orbit() {
        let area = BitMapBackend::new("out.png", (4096, 4096)).into_drawing_area();
        area.fill(&WHITE).unwrap();
        let x_axis = (-1.5..1.5).step(0.1);
        let y_axis = (-1.5..1.5).step(0.1);
        let mut chart = ChartBuilder::on(&area)
            .caption(format!("Grav Test"), ("sans", 20))
            .build_cartesian_2d(x_axis, y_axis)
            .unwrap();

        let grav = gravity(1.0 / 6.649e-11, Vector3::zeros());
        let six_dof = SixDof {
            pos: Vector3::new(1.0, 0., 0.),
            vel: Vector3::new(0.0, 1.0, 0.0),
            mass: 1.0,
            ..SixDof::default()
        };
        let mut six_dof = Sim::new(six_dof);
        six_dof.add_effector(grav);
        let mut time = 0.0;
        let mut points = vec![];
        let dt = 0.01;
        while time <= 2.0 * 3.14 {
            six_dof.tick(dt);
            // six_dof.int_force(time, dt, &grav);
            points.push((six_dof.state.pos.x, six_dof.state.pos.y));
            time += dt;
        }
        chart
            .draw_series(LineSeries::new(points.into_iter(), &BLACK))
            .unwrap();
        area.present().unwrap();
        assert_relative_eq!(six_dof.state.pos, Vector3::new(1.0, 0., 0.), epsilon = 0.01)
    }
}
