use basilisk::att_determination::SunlineConfig;
use basilisk::sys::CSSArraySensorMsgPayload;
use basilisk::sys::NavAttMsgPayload;
use basilisk::{att_determination::SunlineEKF, channel::BskChannel};
use hifitime::Epoch;
use nox::ArrayRepr;
use nox::Tensor;
use nox::Vector;
use nox::MRP;
use nox_frames::earth::ecef_to_eci;
use nox_frames::earth::ned_to_ecef;
use roci::System;
use roci::{Componentize, Decomponentize};
use wmm::GeodeticCoords;

use crate::NavData;

#[derive(Default, Componentize, Decomponentize)]
pub struct World {
    // inputs
    css_inputs: CssInputs,
    #[roci(entity_id = 0, component_id = "mag_ref")]
    mag_ref: Vector<f64, 3, ArrayRepr>,
    #[roci(entity_id = 0, component_id = "sun_ref")]
    sun_ref: Vector<f64, 3, ArrayRepr>,
    #[roci(entity_id = 7, component_id = "mag_value")]
    mag_value: [f64; 3],
    gps_inputs: GpsInputs,

    // outputs
    nav_out: NavData,
}

#[derive(Default, Componentize, Decomponentize)]
struct GpsInputs {
    #[roci(entity_id = 0, component_id = "lat")]
    lat: f64,
    #[roci(entity_id = 0, component_id = "long")]
    long: f64,
    #[roci(entity_id = 0, component_id = "alt")]
    alt: f64,
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

    // magnetic model
    mag_model: wmm::MagneticModel,
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
            mag_model: wmm::MagneticModel::default(),
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
        if let Ok(time) = Epoch::now() {
            world.mag_ref = get_wmm_eci(
                &mut self.mag_model,
                world.gps_inputs.lat,
                world.gps_inputs.long,
                world.gps_inputs.alt,
                time,
            );
            world.sun_ref = nox_frames::earth::sun_vec(time);
        } else {
            println!("failed to get current time");
        }
        let body_1 = Tensor::from_buf(nav_state.vehSunPntBdy);
        let body_2 = Tensor::from_buf(world.mag_value);
        let ref_1 = world.sun_ref;
        let ref_2 = world.mag_ref;
        let att = roci_adcs::triad(body_1, body_2, ref_1, ref_2);
        let att_mrp_bn = MRP::from_rot_matrix(att).0.into_buf();

        world.nav_out.att_mrp_bn = att_mrp_bn;
        world.nav_out.omega_bn_b = nav_state.omega_BN_B;
        world.nav_out.sun_vec_b = nav_state.vehSunPntBdy;
    }
}

fn get_wmm_eci(
    mag_model: &mut wmm::MagneticModel,
    lat: f64,
    long: f64,
    alt: f64,
    time: Epoch,
) -> Vector<f64, 3, ArrayRepr> {
    let coords = GeodeticCoords::with_geoid_height(lat, long, alt); // NOTE: not sure if GPS coords are the geoid height
    let (elements, _) = mag_model.calculate_field(time, coords);
    let field = Vector::<_, 3, ArrayRepr>::new(elements.x, elements.y, elements.z);
    (ecef_to_eci(time) * ned_to_ecef(lat, long)).dot(&field)
}
