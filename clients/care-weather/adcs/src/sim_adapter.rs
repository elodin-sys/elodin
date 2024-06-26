use nox::{ArrayRepr, SpatialTransform};
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
    #[roci(entity_id = 0, component_id = "css_value")]
    pub css_inputs: [f64; 6],
    #[roci(entity_id = 0, component_id = "mag_ref")]
    pub mag_ref: [f64; 3],
    #[roci(entity_id = 0, component_id = "sun_ref")]
    pub sun_ref: [f64; 3],
    #[roci(entity_id = 0, component_id = "mag_value")]
    pub mag_value: [f64; 3],
    #[roci(entity_id = 0, component_id = "mag_postcal_value")]
    pub mag_postcal_value: [f64; 3],
    #[roci(entity_id = 0, component_id = "sun_vec_b")]
    pub sun_vec_b: [f64; 3],
    #[roci(entity_id = 0, component_id = "gyro_omega")]
    pub omega: [f64; 3],
    #[roci(entity_id = 0, component_id = "rw_speed")]
    pub rw_speed: [f64; 3],
    #[roci(entity_id = 0, component_id = "rw_speed_setpoint")]
    pub rw_speed_setpoint: [f64; 3],
}

pub struct SimAdapter<const HZ: usize>;

impl<const HZ: usize> System for SimAdapter<HZ> {
    type World = World;

    type Driver = Hz<HZ>;

    fn update(&mut self, world: &mut Self::World) {
        let mrp = world.nav_out.att_mrp_bn;
        let mrp = nox::MRP::<_, ArrayRepr>(nox::Tensor::from_buf(mrp));
        let quat = nox::Quaternion::from(&mrp);
        world.inertial_pos = SpatialTransform::from_angular(quat);
    }
}
