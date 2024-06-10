use basilisk::att_determination::SunlineConfig;
use basilisk::sys::CSSArraySensorMsgPayload;
use basilisk::sys::NavAttMsgPayload;

use basilisk::{att_determination::SunlineEKF, channel::BskChannel};
use roci::Handler;
use roci::{Componentize, Decomponentize};

use crate::NavData;

#[derive(Default, Componentize, Decomponentize)]
pub struct World {
    nav_out: NavData,

    css_inputs: CssInputs,
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

impl Handler for Determination {
    type World = World;

    fn tick(&mut self, world: &mut Self::World) {
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
        world.nav_out.att_mrp_bn = nav_state.sigma_BN;
        world.nav_out.omega_bn_b = nav_state.omega_BN_B;
        world.nav_out.sun_vec_b = nav_state.vehSunPntBdy;
    }
}
