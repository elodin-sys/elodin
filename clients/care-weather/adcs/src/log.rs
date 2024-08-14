use roci::{Componentize, Decomponentize};
use serde::Serialize;

#[derive(Default, Componentize, Decomponentize, Serialize)]
pub struct World {
    #[roci(entity_id = 0, component_id = "world_pos")]
    inertial_pos: [f64; 7],
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
