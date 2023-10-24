use std::{
    marker::PhantomData,
    ops::{AddAssign, Deref, DerefMut},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};

use crate::{
    builder::{XpbdEffector, XpbdSensor},
    spatial::*,
    tree::Joint,
    FromState,
};
use bevy::prelude::FixedTime;
use bevy_ecs::{
    prelude::{Bundle, Component},
    query::WorldQuery,
    system::Resource,
};
use nalgebra::{matrix, DMatrix, Matrix3, UnitQuaternion, Vector3};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component, Default)]
pub struct Force(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component, Default)]
pub struct Torque(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component)]
pub struct Mass(pub f64);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct JointPos(pub GeneralizedPos);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct JointVel(pub GeneralizedMotion);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Inertia(pub Matrix3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component, Resource)]
pub struct Time(pub f64);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct WorldPos(pub SpatialPos);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct WorldAnchorPos(pub SpatialTransform);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component)]
pub struct WorldVel(pub SpatialMotion);
#[derive(Debug, Clone, PartialEq, Component)]
pub struct SubtreeInertia(pub SpatialInertia);
#[derive(Debug, Clone, PartialEq, Component)]
pub struct TreeIndex(pub usize);
#[derive(Debug, Clone, PartialEq, Resource)]
pub struct TreeMassMatrix(pub DMatrix<f64>);
#[derive(Debug, Clone, PartialEq, Component, Default)]
pub struct JointAccel(pub GeneralizedMotion);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct PrevPos(pub GeneralizedPos);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct InverseInertia(pub Matrix3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct WorldAccel(pub SpatialMotion);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct BiasForce(pub SpatialForce);
#[derive(Debug, Clone, Copy, PartialEq, Component, Default)]
pub struct JointForce(pub GeneralizedForce);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Fixed(pub bool);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Picked(pub bool);
#[derive(Debug, Clone, Copy, PartialEq, Resource)]
pub struct Paused(pub bool);
#[derive(Debug, Resource)]
pub struct PhysicsFixedTime(pub FixedTime);

#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct BodyPos(pub SpatialPos);

impl Inertia {
    pub fn solid_box(width: f64, height: f64, depth: f64, mass: f64) -> Inertia {
        let h = height.powi(2);
        let w = width.powi(2);
        let d = depth.powi(2);
        let k = mass / 12.0;
        Inertia(matrix![
            k * (h + d), 0.0, 0.0;
            0.0, k * ( w + d ), 0.0;
            0.0, 0.0, k * (w + h)
        ])
    }

    pub fn sphere(radius: f64, mass: f64) -> Inertia {
        Inertia(Matrix3::from_diagonal_element(
            2.0 / 5.0 * mass * radius.powi(2),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct SharedNum<T> {
    storage: Arc<AtomicU64>,
    _phantom: PhantomData<T>,
}

impl<T: ToU64Storage + Default> Default for SharedNum<T> {
    fn default() -> Self {
        Self {
            storage: Arc::new(AtomicU64::new(T::default().to_bits())),
            _phantom: Default::default(),
        }
    }
}

impl<T: ToU64Storage> SharedNum<T> {
    pub fn load(&self) -> SharedNumRef<'_, T> {
        let storage = self.storage.load(Ordering::SeqCst);
        SharedNumRef {
            storage: self.storage.as_ref(),
            num: T::from_bits(storage),
        }
    }
}

pub struct SharedNumRef<'a, T: ToU64Storage> {
    storage: &'a AtomicU64,
    num: T,
}

impl<'a, T: ToU64Storage> Deref for SharedNumRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.num
    }
}

impl<'a, T: ToU64Storage> DerefMut for SharedNumRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.num
    }
}

impl<'a, T: ToU64Storage> Drop for SharedNumRef<'a, T> {
    fn drop(&mut self) {
        self.storage.store(self.num.to_bits(), Ordering::SeqCst);
    }
}

pub trait ToU64Storage {
    fn from_bits(num: u64) -> Self;
    fn to_bits(&self) -> u64;
}

impl ToU64Storage for f64 {
    fn from_bits(num: u64) -> Self {
        f64::from_bits(num)
    }

    fn to_bits(&self) -> u64 {
        f64::to_bits(*self)
    }
}

#[derive(Clone, Debug)]
pub struct ObservableNum<T> {
    storage: Arc<ObservableNumInner>,
    _phantom: PhantomData<T>,
}
#[derive(Debug)]
struct ObservableNumInner {
    num: AtomicU64,
    changed: AtomicBool,
}

impl<T: ToU64Storage + Default> Default for ObservableNum<T> {
    fn default() -> Self {
        Self {
            storage: Arc::new(ObservableNumInner {
                num: AtomicU64::new(T::default().to_bits()),
                changed: AtomicBool::new(false),
            }),
            _phantom: Default::default(),
        }
    }
}

impl<T: ToU64Storage> ObservableNum<T> {
    pub fn load(&self) -> ObservableNumRef<'_, T> {
        let storage = self.storage.num.load(Ordering::SeqCst);
        ObservableNumRef {
            storage: self.storage.as_ref(),
            num: T::from_bits(storage),
        }
    }

    pub fn has_changed(&self) -> bool {
        self.storage.changed.swap(false, Ordering::SeqCst)
    }
}

pub struct ObservableNumRef<'a, T: ToU64Storage> {
    storage: &'a ObservableNumInner,
    num: T,
}

impl<'a, T: ToU64Storage> Deref for ObservableNumRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.num
    }
}

impl<'a, T: ToU64Storage> DerefMut for ObservableNumRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.num
    }
}

impl<'a, T: ToU64Storage> Drop for ObservableNumRef<'a, T> {
    fn drop(&mut self) {
        let bits = self.num.to_bits();
        let old = self.storage.num.swap(bits, Ordering::SeqCst);
        self.storage.changed.store(old != bits, Ordering::SeqCst);
    }
}

#[derive(Debug, Resource)]
pub enum TickMode {
    FreeRun,
    Fixed,
    Lockstep(LockStepSignal),
}

#[derive(Debug, Clone, Copy, PartialEq, Resource)]
pub struct Config {
    pub dt: f64,
    pub sub_dt: f64,
    pub substep_count: usize,
    pub scale: f32,
}

impl Default for Config {
    fn default() -> Self {
        let dt = 1.0 / 60.0;
        let substep_count = 24;
        Self {
            dt,
            sub_dt: dt / substep_count as f64,
            substep_count,
            scale: 1.0,
        }
    }
}

#[derive(Component, Default)]
pub struct Effectors(pub Vec<Box<dyn XpbdEffector + Send + Sync>>);

#[derive(Component, Default)]
pub struct Sensors(pub Vec<Box<dyn XpbdSensor + Send + Sync>>);

#[derive(Debug, Clone, Copy, PartialEq, Component, Default)]
pub struct Effect {
    pub force: Force,
    pub torque: Torque,
}

impl Effect {
    pub fn force_at_point(
        r: Vector3<f64>,
        force: Vector3<f64>,
        att: UnitQuaternion<f64>,
    ) -> Effect {
        Effect {
            torque: Torque(att * r.cross(&force)),
            force: Force(att * force),
        }
    }

    pub fn to_spatial(&self, world_pos: Vector3<f64>) -> SpatialForce {
        SpatialForce {
            force: -self.force.0 + world_pos.cross(&self.torque.0),
            torque: -self.torque.0 - world_pos.cross(&self.force.0),
        }
    }
}

impl AddAssign for Effect {
    fn add_assign(&mut self, rhs: Self) {
        self.force.0 += rhs.force.0;
        self.torque.0 += rhs.torque.0;
    }
}

#[derive(Bundle)]
pub struct EntityBundle {
    pub fixed: Fixed,

    // pos
    pub world_pos: WorldPos,
    pub world_anchor_pos: WorldAnchorPos,
    pub world_vel: WorldVel,

    // mass
    pub mass: Mass,
    pub inertia: Inertia,
    pub inverse_inertia: InverseInertia,

    pub effect: Effect,

    pub effectors: Effectors,
    pub sensors: Sensors,

    pub picked: Picked,

    pub joint: JointBundle,

    pub bias_force: BiasForce,
    pub world_accel: WorldAccel,

    pub tree_index: TreeIndex,
    pub subtree_inertia: SubtreeInertia,

    pub body_pos: BodyPos,
}

#[derive(Bundle, Debug)]
pub struct JointBundle {
    pub joint: Joint,
    pub pos: JointPos,
    pub vel: JointVel,
    pub joint_accel: JointAccel,
    pub joint_force: JointForce,
}

#[derive(WorldQuery, Debug)]
#[world_query(mutable, derive(Debug))]
pub struct EntityQuery {
    pub fixed: &'static Fixed,
    pub pos: &'static mut JointPos,
    pub vel: &'static mut JointVel,

    pub world_pos: &'static mut WorldPos,
    pub world_vel: &'static mut WorldVel,

    pub mass: &'static mut Mass,
    pub inertia: &'static mut Inertia,
    pub inverse_inertia: &'static mut InverseInertia,
}

pub struct EntityStateRef<'a> {
    pub fixed: &'a Fixed,
    pub pos: &'a JointPos,
    pub vel: &'a JointVel,

    pub world_pos: &'a WorldPos,
    pub world_vel: &'a WorldVel,

    pub mass: &'a Mass,
    pub inertia: &'a Inertia,
    pub inverse_inertia: &'a InverseInertia,
}

impl<'a> EntityStateRef<'a> {
    pub fn from_query(value: &EntityQueryReadOnlyItem<'a>) -> Self {
        Self {
            pos: value.pos,
            vel: value.vel,
            mass: value.mass,
            inertia: value.inertia,
            inverse_inertia: value.inverse_inertia,
            fixed: value.fixed,
            world_pos: value.world_pos,
            world_vel: value.world_vel,
        }
    }
}

impl<'a> EntityQueryReadOnlyItem<'a> {
    pub fn state_ref(&self) -> EntityStateRef<'_> {
        EntityStateRef::from_query(self)
    }
}

macro_rules! impl_from_state {
    ($state: ty, $component: ty, $field: ident) => {
        impl<'a> FromState<$state> for $component {
            fn from_state(_time: crate::Time, state: &$state) -> Self {
                *state.$field
            }
        }
    };
}

macro_rules! impl_entity_state {
    ($component: ty, $field: ident) => {
        impl_from_state!(EntityStateRef<'a>, $component, $field);
        impl_from_state!(EntityQueryItem<'a>, $component, $field);
        impl_from_state!(EntityQueryReadOnlyItem<'a>, $component, $field);
    };
}

impl_entity_state!(Mass, mass);
impl_entity_state!(JointPos, pos);
impl_entity_state!(WorldPos, world_pos);
impl_entity_state!(JointVel, vel);
impl_entity_state!(Inertia, inertia);
impl_entity_state!(InverseInertia, inverse_inertia);

impl<'a> FromState<EntityStateRef<'a>> for Time {
    fn from_state(time: Time, _state: &EntityStateRef<'a>) -> Self {
        time
    }
}

impl From<Force> for Effect {
    fn from(val: Force) -> Self {
        Effect {
            force: val,
            torque: Torque(Vector3::zeros()),
        }
    }
}

impl From<Torque> for Effect {
    fn from(val: Torque) -> Self {
        Effect {
            torque: val,
            force: Force(Vector3::zeros()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LockStepSignal(Arc<AtomicBool>);

impl LockStepSignal {
    pub fn signal(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn can_continue(&self) -> bool {
        self.0.swap(false, Ordering::Acquire)
    }
}
