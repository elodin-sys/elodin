use basilisk_sys::{
    att_control::MrpPD,
    channel::BskChannel,
    sys::{AttGuidMsgPayload, CmdTorqueBodyMsgPayload},
};
use conduit::{ComponentId, EntityId, Metadata, Query, ValueRepr};
use nox::{ArrayRepr, Quaternion, Scalar, SpatialForce, SpatialTransform, Vector};
use roci::*;
use roci_macros::{Componentize, Decomponentize};
use std::time::Duration;

#[derive(Default, Componentize, Decomponentize)]
struct World {
    #[roci(entity_id = 3, component_id = "world_pos")]
    world_pos: SpatialTransform<f64, ArrayRepr>,
    #[roci(entity_id = 3, component_id = "ang_vel_est")]
    ang_vel_est: Vector<f64, 3, ArrayRepr>,
    #[roci(entity_id = 3, component_id = "control_force")]
    control_force: SpatialForce<f64, ArrayRepr>,
    #[roci(entity_id = 3, component_id = "goal")]
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

impl Handler for MRPHandler {
    type World = World;
    fn tick(&mut self, world: &mut Self::World) {
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
        self.att_guid.write(att_guid_msg_payload);
        self.mrp_pd.update(0);
        let cmd_torque = self.cmd_torque.read();
        world.control_force.inner.inner_mut().buf[..3]
            .copy_from_slice(&cmd_torque.torqueRequestBody);
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    roci::tcp::builder(
        MRPHandler::new(),
        Duration::from_millis(100),
        "127.0.0.1:2242".parse().unwrap(),
    )
    .output(
        Query::with_id(ComponentId::new("control_force")),
        "127.0.0.1:2240".parse().unwrap(),
    )
    .subscribe(
        Query::with_id(ComponentId::new("goal")),
        "127.0.0.1:2240".parse().unwrap(),
    )
    .subscribe(
        Query::with_id(ComponentId::new("world_pos")),
        "127.0.0.1:2240".parse().unwrap(),
    )
    .subscribe(
        Query::with_id(ComponentId::new("ang_vel_est")),
        "127.0.0.1:2240".parse().unwrap(),
    )
    .run()
}
