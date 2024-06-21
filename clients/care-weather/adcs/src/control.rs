use basilisk::{
    att_control::{MRPFeedback, MRPFeedbackConfig, RWArrayConfig, RWAvailability},
    channel::BskChannel,
    effector_interfaces::{RWMotorTorque, RWMotorVoltage, VoltageConfig},
    sys::{
        ArrayMotorTorqueMsgPayload, ArrayMotorVoltageMsgPayload, AttGuidMsgPayload,
        CmdTorqueBodyMsgPayload, RWSpeedMsgPayload, VehicleConfigMsgPayload,
    },
};
use roci::{Componentize, Decomponentize, System};
use serde::{Deserialize, Serialize};

#[derive(Default, Componentize, Decomponentize)]
pub struct World {
    att_err: AttErrInput,
    rw_speed: RWSpeed,
    rw_voltage: RWVoltage,
}
#[derive(Default, Componentize, Decomponentize)]
pub struct RWSpeed {
    #[roci(entity_id = 1, component_id = "rw_speed")]
    rw_speed_0: f64,
    #[roci(entity_id = 2, component_id = "rw_speed")]
    rw_speed_1: f64,
    #[roci(entity_id = 3, component_id = "rw_speed")]
    rw_speed_2: f64,
}

#[derive(Default, Componentize, Decomponentize, Debug)]
pub struct RWVoltage {
    #[roci(entity_id = 1, component_id = "rw_voltage")]
    rw_voltage_0: f64,
    #[roci(entity_id = 2, component_id = "rw_voltage")]
    rw_voltage_1: f64,
    #[roci(entity_id = 3, component_id = "rw_voltage")]
    rw_voltage_2: f64,
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

pub struct Control {
    mrp_feedback: MRPFeedback,
    motor_torque: RWMotorTorque,
    motor_voltage: RWMotorVoltage,

    att_guid_in: BskChannel<AttGuidMsgPayload>,
    rw_speed_in: BskChannel<RWSpeedMsgPayload>,
    motor_voltage_out: BskChannel<ArrayMotorVoltageMsgPayload>,

    #[allow(dead_code)]
    cmd_torque: BskChannel<CmdTorqueBodyMsgPayload>,
    #[allow(dead_code)]
    motor_torque_out: BskChannel<ArrayMotorTorqueMsgPayload>,
}

#[derive(Serialize, Deserialize)]
pub struct ControlConfig {
    control_axes_body: [f64; 9],
    feedback_config: MRPFeedbackConfig,
    rw_config: RWArrayConfig,
    voltage_config: VoltageConfig,
    vehicle_config: VehicleConfig,
}

#[derive(Serialize, Deserialize)]
pub struct VehicleConfig {
    pub spacecraft_inertia_b: [f64; 9usize],
    pub com_b: [f64; 3usize],
    pub spacecraft_mass: f64,
}

impl Control {
    pub fn new(control_config: ControlConfig) -> Self {
        let ControlConfig {
            control_axes_body,
            feedback_config,
            rw_config,
            voltage_config,
            vehicle_config,
        } = control_config;
        let rw_speed_in = BskChannel::pair();
        let rw_array_config = BskChannel::pair();
        let rw_avail = RWAvailability {
            availability: rw_config.wheels.iter().map(|_| true).collect(),
        };
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
        let motor_voltage_out = BskChannel::pair();
        let motor_voltage = RWMotorVoltage::new(
            voltage_config,
            motor_voltage_out.clone(),
            motor_torque_out.clone(),
            rw_speed_in.clone(),
            rw_availability_cmd,
            rw_array_config.clone(),
        );
        Self {
            mrp_feedback,
            motor_torque,
            motor_voltage,
            att_guid_in,
            rw_speed_in,
            motor_voltage_out,
            motor_torque_out,
            cmd_torque,
        }
    }
}

impl System for Control {
    type World = World;
    type Driver = roci::drivers::Hz<100>;

    fn update(&mut self, world: &mut Self::World) {
        let World {
            att_err,
            rw_speed,
            rw_voltage,
        } = world;
        self.att_guid_in.write(
            AttGuidMsgPayload {
                sigma_BR: att_err.att_err_mrp,
                omega_BR_B: att_err.omega_err_br_b,
                omega_RN_B: att_err.omega_rn_b,
                domega_RN_B: att_err.domega_rn_b,
            },
            0,
        );
        let mut wheel_speeds = [0.0; 36];
        wheel_speeds[0] = rw_speed.rw_speed_0;
        wheel_speeds[1] = rw_speed.rw_speed_1;
        wheel_speeds[2] = rw_speed.rw_speed_2;
        self.rw_speed_in.write(
            RWSpeedMsgPayload {
                wheelSpeeds: wheel_speeds,
                wheelThetas: [0.0; 36],
            },
            0,
        );
        self.mrp_feedback.update(0);
        self.motor_torque.update(0);
        self.motor_voltage.update(0);
        let voltage = self.motor_voltage_out.read_msg();
        rw_voltage.rw_voltage_0 = voltage.voltage[0];
        rw_voltage.rw_voltage_1 = voltage.voltage[1];
        rw_voltage.rw_voltage_2 = voltage.voltage[2];
    }
}
