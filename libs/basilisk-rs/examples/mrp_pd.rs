use basilisk::{
    att_control::MrpPD,
    channel::BskChannel,
    sys::{AttGuidMsgPayload, CmdTorqueBodyMsgPayload},
};
use nox::{ArrayRepr, Quaternion, Scalar, SpatialForce, SpatialTransform, Vector};
use roci::{
    drivers::{Driver, Hz, os_sleep_driver},
    *,
};
use roci_macros::{Componentize, Decomponentize};

#[derive(Debug, Default, Componentize, Decomponentize)]
#[roci(parent = "mrp")]
struct World {
    world_pos: SpatialTransform<f64, ArrayRepr>,
    ang_vel_est: Vector<f64, 3, ArrayRepr>,
    control_force: SpatialForce<f64, ArrayRepr>,
    goal: Quaternion<f64, ArrayRepr>,
}

struct MRPHandler {
    mrp_pd: MrpPD,
    att_guid: BskChannel<AttGuidMsgPayload>,
    cmd_torque: BskChannel<CmdTorqueBodyMsgPayload>,
}

fn quat_to_mrp(quat: &Quaternion<f64, ArrayRepr>) -> Vector<f64, 3, ArrayRepr> {
    let w = quat.0.get(3);
    let vec: Vector<f64, 3, ArrayRepr> = quat.0.fixed_slice(&[0]);
    vec / (w + Scalar::from(1.0))
}

impl MRPHandler {
    pub fn new() -> Self {
        let att_guid = BskChannel::pair();
        let vehicle_config = BskChannel::pair();
        let cmd_torque = BskChannel::pair();
        let mrp_pd = MrpPD::new(
            40.0,
            40.0,
            [0.0, 0.0, 0.0],
            vehicle_config,
            att_guid.clone(),
            cmd_torque.clone(),
        );
        Self {
            mrp_pd,
            att_guid,
            cmd_torque,
        }
    }
}

impl System for MRPHandler {
    type World = World;
    type Driver = Hz<10>;
    fn update(&mut self, world: &mut Self::World) {
        let quat = world.world_pos.angular();
        let att_mrp = quat_to_mrp(&quat);
        let goal = quat_to_mrp(&world.goal);
        let ang_vel = &world.ang_vel_est;
        let error_mrp = att_mrp - goal;
        let error_mrp = error_mrp.inner().buf;
        let att_guid_msg_payload = AttGuidMsgPayload {
            sigma_BR: error_mrp,
            omega_BR_B: ang_vel.inner().buf,
            omega_RN_B: [0.0; 3],
            domega_RN_B: [0.0; 3],
        };
        self.att_guid.write(att_guid_msg_payload, 0);
        self.mrp_pd.update(0);
        let cmd_torque = self.cmd_torque.read_msg();
        world.control_force.inner.inner_mut().buf[..3]
            .copy_from_slice(&cmd_torque.torqueRequestBody);
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    os_sleep_driver(MRPHandler::new()).run();
}
