use basilisk::{
    channel::BskChannel,
    sys::{AttGuidMsgPayload, EphemerisMsgPayload, NavAttMsgPayload, NavTransMsgPayload},
};
use nox::{ArrayRepr, SpatialTransform};
use roci::{Componentize, Decomponentize, System};
use serde::{Deserialize, Serialize};

use crate::NavData;

#[derive(Default, Componentize, Decomponentize)]
pub struct World {
    // inputs
    #[roci(entity_id = 0, component_id = "world_pos")]
    inertial_pos: SpatialTransform<f64, ArrayRepr>,
    #[roci(entity_id = 0, component_id = "world_vel")]
    inertial_vel: SpatialTransform<f64, ArrayRepr>,
    nav_in: NavData,

    // outputs
    #[roci(entity_id = 0, component_id = "att_err_mrp")]
    pub att_err_mrp: [f64; 3usize],
    #[roci(entity_id = 0, component_id = "omega_err_br_b")]
    pub omega_err_br_b: [f64; 3usize],
    #[roci(entity_id = 0, component_id = "omega_rn_b")]
    pub omega_rn_b: [f64; 3usize],
    #[roci(entity_id = 0, component_id = "domega_rn_b")]
    pub domega_rn_b: [f64; 3usize],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GuidanceConfig {
    pub sigma_r0r: [f64; 3],
}

pub struct Guidance<const HZ: usize> {
    hill_point: basilisk::att_guidance::HillPoint,
    tracking_error: basilisk::att_guidance::AttTrackingError,

    // inputs
    nav_trans_in: BskChannel<NavTransMsgPayload>,
    celestial_body_in: BskChannel<EphemerisMsgPayload>,
    nav_att_in: BskChannel<NavAttMsgPayload>,

    // outputs
    att_guid_out: BskChannel<AttGuidMsgPayload>,
}

impl<const HZ: usize> Guidance<HZ> {
    pub fn new(sigma_r0r: [f64; 3]) -> Self {
        let att_ref = BskChannel::pair();
        let nav_trans_in = BskChannel::pair();
        let celestial_body_in = BskChannel::pair();
        let att_guid_out = BskChannel::pair();
        let nav_att_in = BskChannel::pair();
        let hill_point = basilisk::att_guidance::HillPoint::new(
            att_ref.clone(),
            nav_trans_in.clone(),
            celestial_body_in.clone(),
        );
        let tracking_error = basilisk::att_guidance::AttTrackingError::new(
            sigma_r0r,
            att_guid_out.clone(),
            nav_att_in.clone(),
            att_ref,
        );
        Self {
            hill_point,
            tracking_error,
            nav_trans_in,
            celestial_body_in,
            att_guid_out,
            nav_att_in,
        }
    }
}

impl<const HZ: usize> System for Guidance<HZ> {
    type World = World;
    type Driver = roci::drivers::Hz<HZ>;

    fn update(&mut self, world: &mut Self::World) {
        // let att = world.inertial_pos.angular().inverse();
        // let ang_vel = att * world.inertial_vel.angular();
        self.nav_att_in.write(
            NavAttMsgPayload {
                timeTag: 0.0,
                sigma_BN: world.nav_in.att_mrp_bn,
                omega_BN_B: world.nav_in.omega_bn_b,
                vehSunPntBdy: world.nav_in.sun_vec_b,
            },
            0,
        );
        self.nav_trans_in.write(
            NavTransMsgPayload {
                timeTag: 0.0,
                r_BN_N: world.inertial_pos.linear().into_buf(),
                v_BN_N: world.inertial_vel.linear().into_buf(),
                vehAccumDV: [0.0; 3],
            },
            0,
        );
        self.celestial_body_in.write(
            EphemerisMsgPayload {
                timeTag: 0.0,
                r_BdyZero_N: [0.0; 3],
                v_BdyZero_N: [0.0; 3],
                sigma_BN: [0.0; 3],
                omega_BN_B: [0.0; 3],
            },
            0,
        );

        self.hill_point.update(0);
        self.tracking_error.update(0);

        let out = self.att_guid_out.read_msg();
        world.att_err_mrp = out.sigma_BR;
        world.omega_err_br_b = out.omega_BR_B;
        world.omega_rn_b = out.omega_RN_B;
        world.domega_rn_b = out.domega_RN_B;
    }
}
