use nox::ArrayRepr;
use nox::SpatialTransform;
use roci::{drivers::Hz, Componentize, Decomponentize, System};

use crate::NavData;

#[derive(Default, Componentize, Decomponentize)]
pub struct World {
    nav_out: NavData,
    #[roci(entity_id = 0, component_id = "world_pos")]
    inertial_pos: SpatialTransform<f64, ArrayRepr>,
}

#[derive(Default, Componentize, Decomponentize)]
pub struct TxWorld {
    // out
    #[roci(entity_id = 0, component_id = "world_pos")]
    inertial_pos: SpatialTransform<f64, ArrayRepr>,
}

pub struct SimAdapter;

impl System for SimAdapter {
    type World = World;

    type Driver = Hz<100>;

    fn update(&mut self, world: &mut Self::World) {
        let mrp = world.nav_out.att_mrp_bn;
        let mrp = nox::MRP::<_, ArrayRepr>(nox::Tensor::from_buf(mrp));
        let quat = nox::Quaternion::from(&mrp);
        world.inertial_pos = SpatialTransform::from_angular(quat);
    }
}
