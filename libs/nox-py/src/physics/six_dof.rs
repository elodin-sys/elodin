use core::ops::{Add, Mul};
use nox::{Op, OwnedRepr, Scalar, SpatialForce, SpatialInertia, SpatialMotion};
use std::sync::Arc;

use crate::ecs::component::WorldPos;
use crate::ecs::{
    Archetype, Component, Query, component_array::ComponentArray, system::IntoSystem,
    system::System,
};
use crate::{
    ErasedSystem,
    physics::integrator::{Integrator, Rk4Ext, semi_implicit_euler, semi_implicit_euler_with_dt},
};

#[derive(Clone)]
pub struct WorldVel<R: OwnedRepr = Op>(pub SpatialMotion<f64, R>);

#[derive(Clone)]
pub struct WorldAccel<R: OwnedRepr = Op>(pub SpatialMotion<f64, R>);

// Manual Component implementations for WorldVel
impl<R: OwnedRepr> impeller2::component::Component for WorldVel<R> {
    const NAME: &'static str = "world_vel";

    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        SpatialMotion::<f64, R>::schema()
    }
}

impl<R: OwnedRepr> Component for WorldVel<R> where
    Self: nox::ReprMonad<Op> + for<'a> nox::FromBuilder<Item<'a> = Self>
{
}

impl<R: OwnedRepr> nox::ReprMonad<R> for WorldVel<R> {
    type Elem = <SpatialMotion<f64, R> as nox::ReprMonad<R>>::Elem;
    type Dim = <SpatialMotion<f64, R> as nox::ReprMonad<R>>::Dim;
    type Map<NewRepr: OwnedRepr> = WorldVel<NewRepr>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        WorldVel(self.0.map(func))
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.0.into_inner()
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        self.0.inner()
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        WorldVel(SpatialMotion::from_inner(inner))
    }
}

// Manual Component implementations for WorldAccel
impl<R: OwnedRepr> impeller2::component::Component for WorldAccel<R> {
    const NAME: &'static str = "world_accel";

    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        SpatialMotion::<f64, R>::schema()
    }
}

impl<R: OwnedRepr> Component for WorldAccel<R> where
    Self: nox::ReprMonad<Op> + for<'a> nox::FromBuilder<Item<'a> = Self>
{
}

impl<R: OwnedRepr> nox::ReprMonad<R> for WorldAccel<R> {
    type Elem = <SpatialMotion<f64, R> as nox::ReprMonad<R>>::Elem;
    type Dim = <SpatialMotion<f64, R> as nox::ReprMonad<R>>::Dim;
    type Map<NewRepr: OwnedRepr> = WorldAccel<NewRepr>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        WorldAccel(self.0.map(func))
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.0.into_inner()
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        self.0.inner()
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        WorldAccel(SpatialMotion::from_inner(inner))
    }
}

// Force and Inertia components
#[derive(Clone)]
pub struct Force<R: OwnedRepr = Op>(pub SpatialForce<f64, R>);

#[derive(Clone)]
pub struct Inertia<R: OwnedRepr = Op>(pub SpatialInertia<f64, R>);

// Manual Component implementations for Force
impl<R: OwnedRepr> impeller2::component::Component for Force<R> {
    const NAME: &'static str = "force";

    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        SpatialForce::<f64, R>::schema()
    }
}

impl<R: OwnedRepr> Component for Force<R> where
    Self: nox::ReprMonad<Op> + for<'a> nox::FromBuilder<Item<'a> = Self>
{
}

impl<R: OwnedRepr> nox::ReprMonad<R> for Force<R> {
    type Elem = <SpatialForce<f64, R> as nox::ReprMonad<R>>::Elem;
    type Dim = <SpatialForce<f64, R> as nox::ReprMonad<R>>::Dim;
    type Map<NewRepr: OwnedRepr> = Force<NewRepr>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        Force(self.0.map(func))
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.0.into_inner()
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        self.0.inner()
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        Force(SpatialForce::from_inner(inner))
    }
}

// Manual Component implementations for Inertia
impl<R: OwnedRepr> impeller2::component::Component for Inertia<R> {
    const NAME: &'static str = "inertia";

    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        SpatialInertia::<f64, R>::schema()
    }
}

impl<R: OwnedRepr> Component for Inertia<R> where
    Self: nox::ReprMonad<Op> + for<'a> nox::FromBuilder<Item<'a> = Self>
{
}

impl<R: OwnedRepr> nox::ReprMonad<R> for Inertia<R> {
    type Elem = <SpatialInertia<f64, R> as nox::ReprMonad<R>>::Elem;
    type Dim = <SpatialInertia<f64, R> as nox::ReprMonad<R>>::Dim;
    type Map<NewRepr: OwnedRepr> = Inertia<NewRepr>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        Inertia(self.0.map(func))
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.0.into_inner()
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        self.0.inner()
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        Inertia(SpatialInertia::from_inner(inner))
    }
}

// Component group definitions for integration
pub struct U {
    pub x: WorldPos,
    pub v: WorldVel,
}

pub struct DU {
    pub v: WorldVel,
    pub a: WorldAccel,
}

// Manual ComponentGroup implementation for U
impl crate::ecs::query::ComponentGroup for U {
    type Params = (Self,);
    type Append<B> = (Self, B);

    fn init(builder: &mut crate::ecs::system::SystemBuilder) -> Result<(), crate::Error> {
        <(WorldPos, WorldVel)>::init(builder)
    }

    fn component_arrays<'a>(
        builder: &'a crate::ecs::system::SystemBuilder,
    ) -> impl Iterator<Item = crate::ecs::component_array::ComponentArray<()>> + 'a {
        <(WorldPos, WorldVel)>::component_arrays(builder)
    }

    fn component_types() -> impl Iterator<Item = elodin_db::ComponentSchema> {
        <(WorldPos, WorldVel)>::component_types()
    }

    fn component_ids() -> impl Iterator<Item = impeller2::types::ComponentId> {
        <(WorldPos, WorldVel)>::component_ids()
    }

    fn component_count() -> usize {
        <(WorldPos, WorldVel)>::component_count()
    }

    fn map_axes() -> &'static [usize] {
        <(WorldPos, WorldVel)>::map_axes()
    }

    fn into_noxpr(self) -> nox::Noxpr {
        nox::Noxpr::tuple(vec![self.x.into_noxpr(), self.v.into_noxpr()])
    }
}

// Manual ComponentGroup implementation for DU
impl crate::ecs::query::ComponentGroup for DU {
    type Params = (Self,);
    type Append<B> = (Self, B);

    fn init(builder: &mut crate::ecs::system::SystemBuilder) -> Result<(), crate::Error> {
        <(WorldVel, WorldAccel)>::init(builder)
    }

    fn component_arrays<'a>(
        builder: &'a crate::ecs::system::SystemBuilder,
    ) -> impl Iterator<Item = crate::ecs::component_array::ComponentArray<()>> + 'a {
        <(WorldVel, WorldAccel)>::component_arrays(builder)
    }

    fn component_types() -> impl Iterator<Item = elodin_db::ComponentSchema> {
        <(WorldVel, WorldAccel)>::component_types()
    }

    fn component_ids() -> impl Iterator<Item = impeller2::types::ComponentId> {
        <(WorldVel, WorldAccel)>::component_ids()
    }

    fn component_count() -> usize {
        <(WorldVel, WorldAccel)>::component_count()
    }

    fn map_axes() -> &'static [usize] {
        <(WorldVel, WorldAccel)>::map_axes()
    }

    fn into_noxpr(self) -> nox::Noxpr {
        nox::Noxpr::tuple(vec![self.v.into_noxpr(), self.a.into_noxpr()])
    }
}

// Manual FromBuilder implementation for U
impl nox::FromBuilder for U {
    type Item<'a> = Self;

    fn from_builder(builder: &nox::Builder) -> Self::Item<'_> {
        Self {
            x: WorldPos::from_builder(builder),
            v: WorldVel::from_builder(builder),
        }
    }
}

// Manual FromBuilder implementation for DU
impl nox::FromBuilder for DU {
    type Item<'a> = Self;

    fn from_builder(builder: &nox::Builder) -> Self::Item<'_> {
        Self {
            v: WorldVel::from_builder(builder),
            a: WorldAccel::from_builder(builder),
        }
    }
}

impl Add<DU> for U {
    type Output = U;

    fn add(self, v: DU) -> Self::Output {
        U {
            x: WorldPos(self.x.0 + v.v.0),
            v: WorldVel(self.v.0 + v.a.0),
        }
    }
}

impl Add for DU {
    type Output = DU;

    fn add(self, v: DU) -> Self::Output {
        DU {
            v: WorldVel(self.v.0 + v.v.0),
            a: WorldAccel(self.a.0 + v.a.0),
        }
    }
}

impl Mul<DU> for Scalar<f64> {
    type Output = DU;

    fn mul(self, rhs: DU) -> Self::Output {
        DU {
            v: WorldVel(&self * rhs.v.0),
            a: WorldAccel(&self * rhs.a.0),
        }
    }
}

impl Mul<DU> for f64 {
    type Output = DU;

    fn mul(self, rhs: DU) -> Self::Output {
        DU {
            v: WorldVel(self * rhs.v.0),
            a: WorldAccel(self * rhs.a.0),
        }
    }
}

impl Add<WorldVel> for WorldPos {
    type Output = WorldPos;

    fn add(self, v: WorldVel) -> Self::Output {
        WorldPos(self.0 + v.0)
    }
}

impl Add<WorldAccel> for WorldVel {
    type Output = WorldVel;

    fn add(self, v: WorldAccel) -> Self::Output {
        WorldVel(self.0 + v.0)
    }
}

impl Mul<WorldVel> for f64 {
    type Output = WorldVel;

    fn mul(self, rhs: WorldVel) -> Self::Output {
        WorldVel(self * rhs.0)
    }
}

impl Mul<WorldVel> for Scalar<f64> {
    type Output = WorldVel;

    fn mul(self, rhs: WorldVel) -> Self::Output {
        WorldVel(&self * rhs.0)
    }
}

impl Mul<WorldAccel> for f64 {
    type Output = WorldAccel;

    fn mul(self, rhs: WorldAccel) -> Self::Output {
        WorldAccel(self * rhs.0)
    }
}

impl Mul<WorldAccel> for Scalar<f64> {
    type Output = WorldAccel;

    fn mul(self, rhs: WorldAccel) -> Self::Output {
        WorldAccel(&self * rhs.0)
    }
}

fn calc_accel(q: Query<(Force, Inertia, WorldPos)>) -> Query<WorldAccel> {
    q.map(|force: Force, inertia: Inertia, pos: WorldPos| {
        let q = pos.0.angular();
        let body_frame_force = q.inverse() * force.0;
        let body_frame_accel = body_frame_force / inertia.0;
        let world_frame_accel = q * body_frame_accel;
        WorldAccel(world_frame_accel)
    })
    .unwrap()
}

fn clear_forces(q: ComponentArray<Force>) -> ComponentArray<Force> {
    q.map(|_| Force(SpatialForce::zero())).unwrap()
}

pub struct Body {
    pub pos: WorldPos,
    pub vel: WorldVel,
    pub accel: WorldAccel,
    pub force: Force,
    pub mass: Inertia,
}

// Manual Archetype implementation for Body
impl Archetype for Body {
    fn components() -> Vec<(
        impeller2::schema::Schema<Vec<u64>>,
        impeller2_wkt::ComponentMetadata,
    )> {
        use crate::ecs::archetype::ComponentExt;
        use impeller2::component::Component as ImpellerComponent;
        vec![
            (<WorldPos>::schema(), <WorldPos>::metadata()),
            (<WorldVel>::schema(), <WorldVel>::metadata()),
            (<WorldAccel>::schema(), <WorldAccel>::metadata()),
            (<Force>::schema(), <Force>::metadata()),
            (<Inertia>::schema(), <Inertia>::metadata()),
        ]
    }

    fn insert_into_world(self, world: &mut crate::ecs::World) {
        self.pos.insert_into_world(world);
        self.vel.insert_into_world(world);
        self.accel.insert_into_world(world);
        self.force.insert_into_world(world);
        self.mass.insert_into_world(world);
    }
}

pub fn six_dof_with_dt<Sys, M, A, R>(
    effectors: impl FnOnce() -> Sys,
    time_step: f64,
    integrator: Integrator,
) -> Arc<dyn System<Arg = (), Ret = ()> + Send + Sync>
where
    M: 'static,
    A: 'static,
    R: 'static,
    Sys: IntoSystem<M, A, R> + 'static,
    <Sys as IntoSystem<M, A, R>>::System: Send + Sync,
{
    let sys = clear_forces.pipe(effectors()).pipe(calc_accel);
    match integrator {
        Integrator::Rk4 => Arc::new(sys.rk4_with_dt::<U, DU>(time_step)),
        Integrator::SemiImplicit => {
            let integrate =
                semi_implicit_euler_with_dt::<WorldPos, WorldVel, WorldAccel>(time_step);
            Arc::new(ErasedSystem::new(sys.pipe(integrate)))
        }
    }
}

pub fn six_dof<Sys, M, A, R>(
    effectors: impl FnOnce() -> Sys,
    integrator: Integrator,
) -> Arc<dyn System<Arg = (), Ret = ()> + Send + Sync>
where
    M: 'static,
    A: 'static,
    R: 'static,
    Sys: IntoSystem<M, A, R> + 'static,
    <Sys as IntoSystem<M, A, R>>::System: Send + Sync,
{
    let sys = clear_forces.pipe(effectors()).pipe(calc_accel);
    match integrator {
        Integrator::Rk4 => Arc::new(sys.rk4::<U, DU>()),
        Integrator::SemiImplicit => {
            let integrate = semi_implicit_euler::<WorldPos, WorldVel, WorldAccel>();
            Arc::new(ErasedSystem::new(sys.pipe(integrate)))
        }
    }
}
