use basilisk::att_determination::SunlineConfig;
use basilisk::sys::CSSArraySensorMsgPayload;
use basilisk::sys::NavAttMsgPayload;
use basilisk::{att_determination::SunlineEKF, channel::BskChannel};
use nox::Tensor;
use nox::MRP;
use roci::System;
use roci::{Componentize, Decomponentize};

use crate::NavData;

#[derive(Default, Componentize, Decomponentize)]
pub struct World {
    // inputs
    css_inputs: CssInputs,
    #[roci(entity_id = 0, component_id = "mag_ref")]
    mag_ref: [f64; 3],
    #[roci(entity_id = 0, component_id = "sun_ref")]
    sun_ref: [f64; 3],
    #[roci(entity_id = 7, component_id = "mag_value")]
    mag_value: [f64; 3],

    // outputs
    nav_out: NavData,
}

#[derive(Default, Componentize, Decomponentize)]
struct CssInputs {
    #[roci(entity_id = 4, component_id = "css_value")]
    css_0: f64,
    #[roci(entity_id = 5, component_id = "css_value")]
    css_1: f64,
    #[roci(entity_id = 6, component_id = "css_value")]
    css_2: f64,
}

pub struct Determination {
    sunline_ekf: SunlineEKF,
    // input
    css_input: BskChannel<CSSArraySensorMsgPayload>,
    // output
    nav_state_out: BskChannel<NavAttMsgPayload>,
}

impl Determination {
    pub fn new(sunline: SunlineConfig) -> Self {
        let nav_state_out = BskChannel::default();
        let filter_data_out = BskChannel::default();
        let css_input = BskChannel::default();
        let sunline_ekf = SunlineEKF::new(
            nav_state_out.clone(),
            filter_data_out,
            css_input.clone(),
            sunline,
        );
        Determination {
            sunline_ekf,
            css_input,
            nav_state_out,
        }
    }
}

impl System for Determination {
    type World = World;
    type Driver = roci::drivers::Hz<100>;

    fn update(&mut self, world: &mut Self::World) {
        let mut css_cos_values = [0.0; 32];
        css_cos_values[0] = world.css_inputs.css_0;
        css_cos_values[1] = world.css_inputs.css_1;
        css_cos_values[2] = world.css_inputs.css_2;
        self.css_input.write(
            CSSArraySensorMsgPayload {
                CosValue: css_cos_values,
            },
            0,
        );
        self.sunline_ekf.update(0);
        let nav_state = self.nav_state_out.read_msg();
        let body_1 = Tensor::from_buf(nav_state.vehSunPntBdy);
        let body_2 = Tensor::from_buf(world.mag_value);
        let ref_1 = Tensor::from_buf(world.sun_ref);
        let ref_2 = Tensor::from_buf(world.mag_ref);
        let att = roci_adcs::triad(body_1, body_2, ref_1, ref_2);
        let att_mrp_bn = MRP::from_rot_matrix(att).0.into_buf();

        world.nav_out.att_mrp_bn = att_mrp_bn;
        world.nav_out.omega_bn_b = nav_state.omega_BN_B;
        world.nav_out.sun_vec_b = nav_state.vehSunPntBdy;
    }
}
