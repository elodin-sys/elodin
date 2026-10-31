use super::set_stage_time;
use crate::World;
use crate::globals::{SimulationTimeStep, StageTime};
use crate::system::{CompiledSystem, System, SystemBuilder, SystemParam};
use crate::{ComponentArray, ComponentGroup, Error, Query};
use nox::Scalar;
use std::ops::Add;
use std::{marker::PhantomData, ops::Mul};

/// Coefficients of an explicit Runge-Kutta method: stage `i` is evaluated `c[i]` of the way
/// through the step, row `i` of `a` weights the derivatives of stages `0..i` in stage `i`,
/// and `b` weights every stage in the step.
pub struct ButcherTableau {
    pub a: &'static [&'static [f64]],
    pub b: &'static [f64],
    pub c: &'static [f64],
}

/// Heun's method (explicit trapezoid rule), second order.
pub const RK2: ButcherTableau = ButcherTableau {
    a: &[&[], &[1.0]],
    b: &[0.5, 0.5],
    c: &[0.0, 1.0],
};

/// Kutta's third-order method.
pub const RK3: ButcherTableau = ButcherTableau {
    a: &[&[], &[0.5], &[-1.0, 2.0]],
    b: &[1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    c: &[0.0, 0.5, 1.0],
};

/// The classic fourth-order Runge-Kutta method.
pub const RK4: ButcherTableau = ButcherTableau {
    a: &[&[], &[0.5], &[0.0, 0.5], &[0.0, 0.0, 1.0]],
    b: &[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
    c: &[0.0, 0.5, 0.5, 1.0],
};

/// Fehlberg's 7(8) pair (NASA TR R-287), advanced with its eighth-order weights.
#[rustfmt::skip]
pub const RK78: ButcherTableau = ButcherTableau {
    a: &[
        &[],
        &[2.0 / 27.0],
        &[1.0 / 36.0, 1.0 / 12.0],
        &[1.0 / 24.0, 0.0, 1.0 / 8.0],
        &[5.0 / 12.0, 0.0, -25.0 / 16.0, 25.0 / 16.0],
        &[1.0 / 20.0, 0.0, 0.0, 1.0 / 4.0, 1.0 / 5.0],
        &[-25.0 / 108.0, 0.0, 0.0, 125.0 / 108.0, -65.0 / 27.0, 125.0 / 54.0],
        &[31.0 / 300.0, 0.0, 0.0, 0.0, 61.0 / 225.0, -2.0 / 9.0, 13.0 / 900.0],
        &[2.0, 0.0, 0.0, -53.0 / 6.0, 704.0 / 45.0, -107.0 / 9.0, 67.0 / 90.0, 3.0],
        &[-91.0 / 108.0, 0.0, 0.0, 23.0 / 108.0, -976.0 / 135.0, 311.0 / 54.0, -19.0 / 60.0,
          17.0 / 6.0, -1.0 / 12.0],
        &[2383.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0, -301.0 / 82.0,
          2133.0 / 4100.0, 45.0 / 82.0, 45.0 / 164.0, 18.0 / 41.0],
        &[3.0 / 205.0, 0.0, 0.0, 0.0, 0.0, -6.0 / 41.0, -3.0 / 205.0, -3.0 / 41.0, 3.0 / 41.0,
          6.0 / 41.0, 0.0],
        &[-1777.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0, -289.0 / 82.0,
          2193.0 / 4100.0, 51.0 / 82.0, 33.0 / 164.0, 12.0 / 41.0, 0.0, 1.0],
    ],
    b: &[0.0, 0.0, 0.0, 0.0, 0.0, 34.0 / 105.0, 9.0 / 35.0, 9.0 / 35.0, 9.0 / 280.0, 9.0 / 280.0,
         0.0, 41.0 / 840.0, 41.0 / 840.0],
    c: &[0.0, 2.0 / 27.0, 1.0 / 9.0, 1.0 / 6.0, 5.0 / 12.0, 0.5, 5.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0,
         1.0 / 3.0, 1.0, 0.0, 1.0],
};

/// Integrates state `U` with an explicit Runge-Kutta method. After each evaluation of
/// `pipe`, the components in `S` are read and converted into the stage derivative `DU`.
pub struct ExplicitRk<U, S, DU, Pipe> {
    tableau: &'static ButcherTableau,
    dt: Option<f64>,
    pipe: Pipe,
    phantom_data: PhantomData<(U, S, DU)>,
}

impl<Pipe, U, S, DU> ExplicitRk<U, S, DU, Pipe> {
    pub fn new(tableau: &'static ButcherTableau, pipe: Pipe, dt: Option<f64>) -> Self {
        Self {
            tableau,
            dt,
            pipe,
            phantom_data: PhantomData,
        }
    }
}

pub trait Rk4Ext {
    fn rk4<U, DU>(self) -> ExplicitRk<U, DU, DU, Self>
    where
        Self: Sized;
    fn rk4_with_dt<U, DU>(self, dt: f64) -> ExplicitRk<U, DU, DU, Self>
    where
        Self: Sized;
}

impl<Sys> Rk4Ext for Sys
where
    Sys: System,
{
    fn rk4<U, DU>(self) -> ExplicitRk<U, DU, DU, Self>
    where
        Self: Sized,
    {
        ExplicitRk::new(&RK4, self, None)
    }

    fn rk4_with_dt<U, DU>(self, dt: f64) -> ExplicitRk<U, DU, DU, Self>
    where
        Self: Sized,
    {
        ExplicitRk::new(&RK4, self, Some(dt))
    }
}

impl<Pipe, U, S, DU> System for ExplicitRk<U, S, DU, Pipe>
where
    Query<U>: SystemParam<Item = Query<U>> + Clone,
    Query<(S,)>: SystemParam<Item = Query<(S,)>>,
    U: Add<DU, Output = U> + ComponentGroup + for<'a> nox::FromBuilder<Item<'a> = U> + Send + Sync,
    S: ComponentGroup + for<'a> nox::FromBuilder<Item<'a> = S> + Send + Sync,
    DU: From<S>
        + Add<DU, Output = DU>
        + ComponentGroup
        + for<'a> nox::FromBuilder<Item<'a> = DU>
        + Send
        + Sync,
    f64: Mul<DU, Output = DU>,
    Scalar<f64>: Mul<DU, Output = DU>,
    Pipe: System + Send + Sync,
{
    type Arg = ();
    type Ret = ();

    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
        self.pipe.init(builder)?;
        ComponentArray::<SimulationTimeStep>::init(builder)?;
        ComponentArray::<StageTime>::init(builder)?;
        Query::<U>::init(builder)?;
        Query::<S>::init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
        let mut builder = SystemBuilder::new(world);
        let compiled_pipe = self.pipe.compile(world)?;
        self.init(&mut builder)?;
        let init_u = Query::<U>::param(&builder)?;
        let sim_dt = ComponentArray::<SimulationTimeStep>::param(&builder)?;
        let dt = self.dt.map(Scalar::from).unwrap_or_else(|| sim_dt.get(0).0);
        let stage_time = ComponentArray::<StageTime>::param(&builder)?;

        // init_u + dt * Σ w_j k_j. U and S can share components, so states are built from
        // the init_u and k_j snapshots rather than read back from the builder.
        let advance = |w: &[f64], k: &[Query<DU>]| -> Result<Query<U>, Error> {
            let mut terms = w.iter().zip(k).filter(|(w, _)| **w != 0.0);
            let Some((&w0, k0)) = terms.next() else {
                return Ok(init_u.clone());
            };
            let mut du = k0.clone().transmute::<(DU,)>().map(move |k: DU| w0 * k)?;
            for (&w, k) in terms {
                du = du
                    .join_query(k.clone())
                    .map(move |du: DU, k: DU| du + w * k)?;
            }
            init_u
                .clone()
                .join_query(du)
                .map(|u, du| u + dt.clone() * du)
        };

        let mut k = Vec::with_capacity(self.tableau.b.len());
        for (a, &c) in self.tableau.a.iter().zip(self.tableau.c) {
            advance(a, &k)?.insert_into_builder(&mut builder);
            set_stage_time(&stage_time, &dt * c, &mut builder)?;
            compiled_pipe.clone().insert_into_builder(&mut builder)?;
            k.push(Query::<(S,)>::param(&builder)?.map(|s: S| DU::from(s))?);
        }
        advance(self.tableau.b, &k)?.insert_into_builder(&mut builder);
        Query::from(stage_time).insert_into_builder(&mut builder);
        builder.to_compiled_system()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use elodin_macros::{Archetype, Component, ReprMonad};
    use impeller::component::Component as ComponentTrait;
    use nox::{Op, OwnedRepr};

    #[derive(Clone, Component, ReprMonad)]
    struct X<R: OwnedRepr = Op>(Scalar<f64, R>);

    #[derive(Clone, Component, ReprMonad)]
    struct V<R: OwnedRepr = Op>(Scalar<f64, R>);

    impl Add<V> for X {
        type Output = X;
        fn add(self, v: V) -> X {
            X(self.0 + v.0)
        }
    }

    impl Add for V {
        type Output = V;
        fn add(self, rhs: V) -> V {
            V(self.0 + rhs.0)
        }
    }

    impl Mul<V> for f64 {
        type Output = V;
        fn mul(self, rhs: V) -> V {
            V(self * rhs.0)
        }
    }

    impl Mul<V> for Scalar<f64> {
        type Output = V;
        fn mul(self, rhs: V) -> V {
            V(self * rhs.0)
        }
    }

    #[derive(Archetype)]
    struct Body {
        x: X,
        v: V,
    }

    #[test]
    fn rk4_pipeline_shape() {
        let mut world = World::default();
        world.spawn(Body {
            x: X(0.0.into()),
            v: V(10.0.into()),
        });

        let compiled = ().rk4::<X, V>().compile(&world).unwrap();

        let x_id = <X<Op> as ComponentTrait>::COMPONENT_ID;
        let v_id = <V<Op> as ComponentTrait>::COMPONENT_ID;
        let dt_id = <SimulationTimeStep<Op> as ComponentTrait>::COMPONENT_ID;

        assert!(compiled.inputs.contains(&x_id), "X missing from inputs");
        assert!(compiled.inputs.contains(&v_id), "V missing from inputs");
        assert!(
            compiled.inputs.contains(&dt_id),
            "SimulationTimeStep missing from inputs"
        );
        assert!(compiled.outputs.contains(&x_id), "X missing from outputs");

        let x_slot = compiled
            .input_slots
            .iter()
            .find(|s| s.component_id == x_id)
            .expect("X input slot");
        // Scalar component with a singleton column: zero-dim (batch axis elided).
        assert!(x_slot.shape.is_empty(), "X slot shape: {:?}", x_slot.shape);
        assert!(x_slot.entity_axis_elided);

        let v_slot = compiled
            .input_slots
            .iter()
            .find(|s| s.component_id == v_id)
            .expect("V input slot");
        assert!(v_slot.shape.is_empty(), "V slot shape: {:?}", v_slot.shape);
        assert!(v_slot.entity_axis_elided);

        // The inner computation function's arity must agree with the number of
        // declared inputs - a structural invariant of `SystemBuilder`.
        assert_eq!(compiled.computation.func.args.len(), compiled.inputs.len());
    }

    #[test]
    fn tableaux_are_consistent() {
        for t in [&RK2, &RK3, &RK4, &RK78] {
            assert_eq!((t.a.len(), t.c.len()), (t.b.len(), t.b.len()));
            for (i, (row, c)) in t.a.iter().zip(t.c).enumerate() {
                assert_eq!(row.len(), i);
                assert!((row.iter().sum::<f64>() - c).abs() < 1e-14, "row {i}");
            }
            assert!((t.b.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        }
    }
}
