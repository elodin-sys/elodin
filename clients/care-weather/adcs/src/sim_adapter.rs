use nox::ArrayRepr;
use nox::SpatialMotion;
use nox::SpatialTransform;
use roci::{drivers::Hz, Componentize, Decomponentize, System};

use crate::NavData;

#[derive(Default, Componentize, Decomponentize)]
pub struct World {
    // in
    #[roci(entity_id = 0, component_id = "world_pos")]
    inertial_pos: SpatialTransform<f64, ArrayRepr>,
    #[roci(entity_id = 0, component_id = "world_vel")]
    inertial_vel: SpatialMotion<f64, ArrayRepr>,

    // out
    nav_out: NavData,
}
pub struct SimAdapter;

impl System for SimAdapter {
    type World = World;

    type Driver = Hz<100>;

    fn update(&mut self, world: &mut Self::World) {
        //println!("world.inertial_pos: {:?}", world.inertial_pos);
        world.nav_out.att_mrp_bn = dbg!(world.inertial_pos.angular().mrp().0.into_buf());
        world.nav_out.omega_bn_b = (world.inertial_vel.angular()).into_buf();
    }
}
