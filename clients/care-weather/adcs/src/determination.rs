use std::time::Instant;

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
use roci_adcs::mekf;
use serde::Deserialize;
use serde::Serialize;
use wmm::GeodeticCoords;

use crate::NavData;

#[derive(Default, Componentize, Decomponentize)]
pub struct World {
    // inputs
    #[roci(entity_id = 0, component_id = "css_value")]
    pub css_inputs: [f64; 6],
    #[roci(entity_id = 0, component_id = "mag_ref")]
    pub mag_ref: Vector<f64, 3, ArrayRepr>,
    #[roci(entity_id = 0, component_id = "sun_ref")]
    pub sun_ref: Vector<f64, 3, ArrayRepr>,
    #[roci(entity_id = 0, component_id = "mag_value")]
    pub mag_value: [f64; 3],
    #[roci(entity_id = 0, component_id = "mag_postcal_value")]
    pub mag_postcal_value: [f64; 3],
    pub gps_inputs: GpsInputs,
    #[roci(entity_id = 0, component_id = "gyro_omega")]
    pub omega: Vector<f64, 3, ArrayRepr>,

    // outputs
    pub nav_out: NavData,
}

#[derive(Debug, Default, Componentize, Decomponentize)]
pub struct GpsInputs {
    #[roci(entity_id = 0, component_id = "lat")]
    pub lat: f64,
    #[roci(entity_id = 0, component_id = "long")]
    pub long: f64,
    #[roci(entity_id = 0, component_id = "alt")]
    pub alt: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MEKFConfig {
    sigma_g: [f64; 3],
    sigma_b: [f64; 3],
    dt: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MagCal {
    pub t: [[f64; 3]; 3],
    pub h: [f64; 3],
}

impl Default for MagCal {
    fn default() -> Self {
        MagCal {
            t: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            h: [0.0; 3],
        }
    }
}

pub struct Determination {
    start: Instant,
    sunline_ekf: SunlineEKF,
    // input
    css_input: BskChannel<CSSArraySensorMsgPayload>,
    // output
    nav_state_out: BskChannel<NavAttMsgPayload>,

    // magnetic model
    mag_model: wmm::MagneticModel,

    mekf_state: Option<mekf::State>,
    mag_cal: MagCal,
}

impl Determination {
    pub fn new(sunline: SunlineConfig, mag_cal: MagCal, mekf_config: Option<MEKFConfig>) -> Self {
        let nav_state_out = BskChannel::default();
        let filter_data_out = BskChannel::default();
        let css_input = BskChannel::default();
        let sunline_ekf = SunlineEKF::new(
            nav_state_out.clone(),
            filter_data_out,
            css_input.clone(),
            sunline,
        );
        let mekf_state = mekf_config.map(|config| {
            mekf::State::new(
                Tensor::from_buf(config.sigma_g),
                Tensor::from_buf(config.sigma_b),
                config.dt,
            )
        });
        Determination {
            start: Instant::now(),
            sunline_ekf,
            css_input,
            nav_state_out,
            mag_model: wmm::MagneticModel::default(),
            mekf_state,
            mag_cal,
        }
    }
}

impl System for Determination {
    type World = World;
    type Driver = roci::drivers::Hz<100>;

    fn update(&mut self, world: &mut Self::World) {
        let elapsed = self.start.elapsed().as_nanos() as u64;
        let mut css_cos_values = [0.0; 32];
        css_cos_values[..world.css_inputs.len()].copy_from_slice(&world.css_inputs);
        self.css_input.write(
            CSSArraySensorMsgPayload {
                CosValue: css_cos_values,
            },
            elapsed,
        );
        self.sunline_ekf.update(elapsed);
        let nav_state = self.nav_state_out.read_msg();
        if let Ok(time) = Epoch::now() {
            world.mag_ref = get_wmm_eci(
                &mut self.mag_model,
                world.gps_inputs.lat,
                world.gps_inputs.long,
                world.gps_inputs.alt,
                time,
            )
            .normalize();
            world.sun_ref = nox_frames::earth::sun_vec(time).normalize();
        } else {
            println!("failed to get current time");
        }
        let body_1 = Tensor::from_buf(nav_state.vehSunPntBdy);
        let hard_iron_cal: Tensor<f64, _, _> =
            Vector::from_buf(world.mag_value) - Vector::from_buf(self.mag_cal.h);
        let body_2 = hard_iron_cal.normalize();
        // let soft_iron_cal = nox::Matrix3::from_buf(self.mag_cal.t).dot(&hard_iron_cal);
        // let body_2 = soft_iron_cal;
        world.mag_postcal_value = body_2.into_buf();
        let ref_1 = world.sun_ref;
        let ref_2 = world.mag_ref;
        let att_mrp_bn = if let Some(mut mekf_state) = self.mekf_state.take() {
            mekf_state.omega = world.omega;
            let mekf_state = mekf_state.estimate_attitude([body_1, body_2], [ref_1, ref_2]);
            let mekf_state = self.mekf_state.insert(mekf_state);
            MRP::from(mekf_state.q_hat).0.into_buf()
        } else {
            let att = roci_adcs::triad(body_1, body_2, ref_1, ref_2);
            MRP::from_rot_matrix(att).0.into_buf()
        };

        // println!("css_inp: {:?}", world.css_inputs);
        // println!("sun_val: {:?}", nav_state.vehSunPntBdy);
        // println!("mag_val: {:?}", world.mag_value);
        // println!("sun_ref: {:?}", world.sun_ref);
        // println!("mag_ref: {:?}", world.mag_ref);
        // println!("att_mrp: {:?}", att_mrp_bn);

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
