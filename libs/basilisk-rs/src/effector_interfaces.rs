use crate::{
    channel::BskChannel,
    sys::{
        rwMotorTorqueConfig, rwMotorVoltageConfig, ArrayMotorTorqueMsgPayload,
        ArrayMotorVoltageMsgPayload, CmdTorqueBodyMsgPayload, RWArrayConfigMsgPayload,
        RWAvailabilityMsgPayload, RWSpeedMsgPayload,
    },
};

pub const MODULE_ID: i64 = 0x224E;

pub struct RWMotorTorque {
    config: rwMotorTorqueConfig,
}

impl RWMotorTorque {
    pub fn new(
        control_axes_body: [f64; 9],
        motor_torque_out: BskChannel<ArrayMotorTorqueMsgPayload>,
        cmd_torque_in: BskChannel<CmdTorqueBodyMsgPayload>,
        rw_params: BskChannel<RWArrayConfigMsgPayload>,
        rw_availability: BskChannel<RWAvailabilityMsgPayload>,
    ) -> Self {
        Self {
            config: rwMotorTorqueConfig {
                controlAxes_B: control_axes_body,
                numControlAxes: 0,
                numAvailRW: 0,
                rwConfigParams: Default::default(),
                GsMatrix_B: [0.0; 108],
                CGs: [[0.0; 36]; 3],
                rwMotorTorqueOutMsg: motor_torque_out.into(),
                vehControlInMsg: cmd_torque_in.into(),
                rwParamsInMsg: rw_params.into(),
                rwAvailInMsg: rw_availability.into(),
                bskLogger: std::ptr::null_mut(),
            },
        }
    }

    pub fn reset(&mut self) {
        unsafe { crate::sys::Reset_rwMotorTorque(&mut self.config, 0, MODULE_ID) }
    }
    pub fn update(&mut self, time: u64) {
        unsafe { crate::sys::Update_rwMotorTorque(&mut self.config, time, MODULE_ID) }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VoltageConfig {
    pub voltage_max: f64,
    pub voltage_min: f64,
    pub k: f64,
}

pub struct RWMotorVoltage {
    config: rwMotorVoltageConfig,
}

impl RWMotorVoltage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: VoltageConfig,
        motor_voltage_out: BskChannel<ArrayMotorVoltageMsgPayload>,
        torque_in: BskChannel<ArrayMotorTorqueMsgPayload>,
        speed_in: BskChannel<RWSpeedMsgPayload>,
        availability_in: BskChannel<RWAvailabilityMsgPayload>,
        rw_config: BskChannel<RWArrayConfigMsgPayload>,
    ) -> Self {
        let VoltageConfig {
            voltage_max,
            voltage_min,
            k,
        } = config;
        let mut this = Self {
            config: rwMotorVoltageConfig {
                VMin: voltage_max,
                VMax: voltage_min,
                voltageOutMsg: motor_voltage_out.into(),
                torqueInMsg: torque_in.into(),
                rwParamsInMsg: rw_config.into(),
                rwSpeedInMsg: speed_in.into(),
                rwAvailInMsg: availability_in.into(),
                bskLogger: std::ptr::null_mut(),
                K: k,
                rwSpeedOld: [0.0; 36],
                priorTime: 0,
                resetFlag: 1,
                rwConfigParams: Default::default(),
            },
        };
        this.reset();
        this
    }

    pub fn reset(&mut self) {
        unsafe { crate::sys::Reset_rwMotorVoltage(&mut self.config, 0, MODULE_ID) }
    }

    pub fn update(&mut self, time: u64) {
        unsafe { crate::sys::Update_rwMotorVoltage(&mut self.config, time, MODULE_ID) }
    }
}
