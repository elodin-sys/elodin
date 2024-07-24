use basilisk::{
    att_control::{MRPFeedback, MRPFeedbackConfig, RWArrayConfig, RWAvailability},
    channel::BskChannel,
    effector_interfaces::RWMotorTorque,
    sys::{
        ArrayMotorTorqueMsgPayload, AttGuidMsgPayload, CmdTorqueBodyMsgPayload, RWSpeedMsgPayload,
        VehicleConfigMsgPayload,
    },
};
use nox::{ArrayRepr, Quaternion, Vector};
use roci::{Componentize, Decomponentize, System};
use serde::{Deserialize, Serialize};

use crate::NavData;

#[derive(Default, Componentize, Decomponentize)]
pub struct World {
    att_err: AttErrInput,
    // inputs
    #[roci(entity_id = 0, component_id = "rw_speed")]
    rw_speed: Vector<f64, 3, ArrayRepr>,

    // hacks inputs
    #[roci(entity_id = 0, component_id = "target_att")]
    target_att: [f64; 3],

    // outputs
    #[roci(entity_id = 0, component_id = "rw_torque")]
    rw_torque: Vector<f64, 3, ArrayRepr>,
    #[roci(entity_id = 0, component_id = "rw_pwm_setpoint")]
    rw_pwm_setpoint: Vector<f64, 3, ArrayRepr>,

    nav_out: NavData,
}

#[derive(Default, Componentize, Decomponentize, Debug)]
pub struct AttErrInput {
    #[roci(entity_id = 0, component_id = "att_err_mrp")]
    pub att_err_mrp: [f64; 3usize],
    #[roci(entity_id = 0, component_id = "omega_err_br_b")]
    pub omega_err_br_b: [f64; 3usize],
    #[roci(entity_id = 0, component_id = "omega_rn_b")]
    pub omega_rn_b: [f64; 3usize],
    #[roci(entity_id = 0, component_id = "domega_rn_b")]
    pub domega_rn_b: [f64; 3usize],
}

pub struct Control<const HZ: usize> {
    mrp_feedback: MRPFeedback,
    motor_torque: RWMotorTorque,

    att_guid_in: BskChannel<AttGuidMsgPayload>,
    rw_speed_in: BskChannel<RWSpeedMsgPayload>,

    #[allow(dead_code)]
    cmd_torque: BskChannel<CmdTorqueBodyMsgPayload>,
    #[allow(dead_code)]
    motor_torque_out: BskChannel<ArrayMotorTorqueMsgPayload>,
    rw_max_torque: Vector<f64, 3, ArrayRepr>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ControlConfig {
    control_axes_body: [f64; 9],
    feedback_config: MRPFeedbackConfig,
    rw_config: RWArrayConfig,
    vehicle_config: VehicleConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VehicleConfig {
    pub spacecraft_inertia_b: [f64; 9usize],
    pub com_b: [f64; 3usize],
    pub spacecraft_mass: f64,
}

impl<const HZ: usize> Control<HZ> {
    pub fn new(control_config: ControlConfig) -> Self {
        let ControlConfig {
            control_axes_body,
            feedback_config,
            rw_config,
            vehicle_config,
        } = control_config;
        let rw_speed_in = BskChannel::pair();
        let rw_array_config = BskChannel::pair();
        let rw_avail = RWAvailability {
            availability: rw_config.wheels.iter().map(|_| true).collect(),
        };

        let mut rw_max_torques = [0.0; 3];
        rw_config
            .wheels
            .iter()
            .zip(rw_max_torques.iter_mut())
            .for_each(|(wheel, out)| {
                *out = wheel.max_torque;
            });
        let rw_max_torque = Vector::from_buf(rw_max_torques);

        rw_array_config.write(rw_config.into(), 0);
        let rw_availability_cmd = BskChannel::pair();
        rw_availability_cmd.write(rw_avail.into(), 0);
        let cmd_torque = BskChannel::pair();
        let int_feedback = BskChannel::pair();
        let att_guid_in = BskChannel::pair();
        let vehicle_config_channel = BskChannel::pair();
        vehicle_config_channel.write(
            VehicleConfigMsgPayload {
                ISCPntB_B: vehicle_config.spacecraft_inertia_b,
                CoM_B: vehicle_config.com_b,
                massSC: vehicle_config.spacecraft_mass,
                CurrentADCSState: 0,
            },
            0,
        );
        let mrp_feedback = MRPFeedback::new(
            feedback_config,
            rw_speed_in.clone(),
            rw_availability_cmd.clone(),
            rw_array_config.clone(),
            cmd_torque.clone(),
            int_feedback,
            att_guid_in.clone(),
            vehicle_config_channel,
        );
        let motor_torque_out = BskChannel::pair();
        let motor_torque = RWMotorTorque::new(
            control_axes_body,
            motor_torque_out.clone(),
            cmd_torque.clone(),
            rw_array_config.clone(),
            rw_availability_cmd.clone(),
        );
        Self {
            mrp_feedback,
            motor_torque,
            att_guid_in,
            rw_speed_in,
            motor_torque_out,
            cmd_torque,
            rw_max_torque,
        }
    }
}

impl<const HZ: usize> System for Control<HZ> {
    type World = World;
    type Driver = roci::drivers::Hz<HZ>;

    fn update(&mut self, world: &mut Self::World) {
        let World {
            att_err,
            rw_speed,
            rw_torque,
            rw_pwm_setpoint,
            nav_out,
            target_att,
        } = world;
        let mrp = nav_out.att_mrp_bn;
        let mrp = nox::MRP::<f64, ArrayRepr>(nox::Tensor::from_buf(mrp));
        let quat = nox::Quaternion::from(&mrp);

        let target_q = Quaternion::from_euler(Vector::from_buf(*target_att));

        //let target_q = Quaternion::<f64, ArrayRepr>::from_axis_angle(Vector::z_axis(), PI / 2.0);
        let error = quat * target_q.inverse();
        // dbg!(error);
        let [_, _, z] = error.mrp().0.into_buf();
        let [x, y, z_omega] = nav_out.omega_bn_b;
        self.att_guid_in.write(
            AttGuidMsgPayload {
                sigma_BR: [0.0, 0.0, z],
                omega_BR_B: [x, y, z_omega],
                omega_RN_B: att_err.omega_rn_b,
                domega_RN_B: att_err.domega_rn_b,
            },
            0,
        );
        let mut wheel_speeds = [0.0; 36];
        wheel_speeds[..3].copy_from_slice(&rw_speed.into_buf());
        self.rw_speed_in.write(
            RWSpeedMsgPayload {
                wheelSpeeds: wheel_speeds,
                wheelThetas: [0.0; 36],
            },
            0,
        );
        self.mrp_feedback.update(0);
        self.motor_torque.update(0);
        let motor_torques = self.motor_torque_out.read_msg().motorTorque;

        // clamp rw_torque to max_torque
        let mut torques = [0.0; 3];
        torques
            .iter_mut()
            .zip(motor_torques)
            .zip(self.rw_max_torque.into_buf())
            .for_each(|((torque, rw_torque), max_torque)| {
                *torque = rw_torque.clamp(-max_torque, max_torque);
            });
        *rw_torque = Vector::from_buf(torques);

        *rw_pwm_setpoint = (*rw_torque / self.rw_max_torque) * 4096.0;
    }
}
