use crate::{
    channel::BskChannel,
    sys::{
        AttGuidMsgPayload, CmdTorqueBodyMsgPayload, MrpPDConfig, RWArrayConfigMsgPayload,
        RWAvailabilityMsgPayload, RWSpeedMsgPayload, RateCmdMsgPayload, Reset_mrpPD, Update_mrpPD,
        VehicleConfigMsgPayload, mrpFeedbackConfig, mrpSteeringConfig,
    },
};

pub const MODULE_ID: i64 = 0x224A;

/// MRP steering control module - a wrapper around `mrpSteering`
///
/// See https://hanspeterschaub.info/basilisk/Documentation/fswAlgorithms/attControl/mrpSteering/mrpSteering.html for more info
pub struct MRPSteering {
    config: mrpSteeringConfig,
}

impl MRPSteering {
    /// Create a new MRPSteering
    ///
    /// # Args
    /// - `k1`: proportional gain (rad/s)
    /// - `k3`: cubic gained applied to mrp error in steering saturation function (rad/s)
    /// - omega_max: maximum command rate (rad/s)
    /// - feed_forward_enable: if outer feed forward is enabled
    /// - rate_cmd: channel for rate command
    /// - att_guid_cmd: channel for attitude guidance command
    pub fn new(
        k1: f64,
        k3: f64,
        omega_max: f64,
        feed_forward_enable: bool,
        rate_cmd: BskChannel<RateCmdMsgPayload>,
        att_guid_cmd: BskChannel<AttGuidMsgPayload>,
    ) -> Self {
        let config = mrpSteeringConfig {
            K1: k1,
            K3: k3,
            omega_max,
            ignoreOuterLoopFeedforward: if feed_forward_enable { 0 } else { 1 },
            rateCmdOutMsg: rate_cmd.into(),
            guidInMsg: att_guid_cmd.into(),
            bskLogger: std::ptr::null_mut(),
        };
        let mut this = Self { config };
        this.reset();
        this
    }

    /// Resets the MRPSteering module
    pub fn reset(&mut self) {
        unsafe {
            crate::sys::Reset_mrpSteering(&mut self.config, 0, MODULE_ID);
        }
    }

    /// Ticks the mrpSteering module forward, pulling any new messages, and sending output message
    pub fn update(&mut self, time: u64) {
        unsafe {
            crate::sys::Update_mrpSteering(&mut self.config, time, MODULE_ID);
        }
    }
}

/// Enum for MRPFeedback's two possible control laws
#[cfg_attr(feature = "serde", derive(Debug, serde::Serialize, serde::Deserialize))]
#[derive(Default)]
#[repr(i32)]
pub enum MRPFeedbackControlLaw {
    #[default]
    A = 0,
    B = 1,
}

pub struct MRPFeedback {
    config: mrpFeedbackConfig,
}

#[cfg_attr(feature = "serde", derive(Debug, serde::Serialize, serde::Deserialize))]
pub struct MRPFeedbackConfig {
    k: f64,
    p: f64,
    k_i: f64,
    integral_limit: f64,
    control_law: MRPFeedbackControlLaw,
    prior_time: u64,
    z: [f64; 3],
    int_sigma: [f64; 3],
    known_torque_pnt_b_b: [f64; 3],
    inertia: [f64; 9],
}

impl MRPFeedback {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: MRPFeedbackConfig,
        rw_speed_cmd: BskChannel<RWSpeedMsgPayload>,
        rw_availability_cmd: BskChannel<RWAvailabilityMsgPayload>,
        rw_array_config: BskChannel<RWArrayConfigMsgPayload>,
        cmd_torque_body: BskChannel<CmdTorqueBodyMsgPayload>,
        int_feedback: BskChannel<CmdTorqueBodyMsgPayload>,
        att_guid_cmd: BskChannel<AttGuidMsgPayload>,
        vehicle_config: BskChannel<VehicleConfigMsgPayload>,
    ) -> Self {
        let MRPFeedbackConfig {
            k,
            p,
            k_i,
            integral_limit,
            control_law,
            prior_time,
            z,
            int_sigma,
            known_torque_pnt_b_b,
            inertia,
        } = config;
        let config = mrpFeedbackConfig {
            K: k,
            P: p,
            Ki: k_i,
            integralLimit: integral_limit,
            controlLawType: control_law as i32,
            priorTime: prior_time,
            z,
            int_sigma,
            ISCPntB_B: inertia,
            knownTorquePntB_B: known_torque_pnt_b_b,
            rwSpeedsInMsg: rw_speed_cmd.into(),
            rwAvailInMsg: rw_availability_cmd.into(),
            rwParamsInMsg: rw_array_config.into(),
            intFeedbackTorqueOutMsg: int_feedback.into(),
            cmdTorqueOutMsg: cmd_torque_body.into(),
            guidInMsg: att_guid_cmd.into(),
            vehConfigInMsg: vehicle_config.into(),
            bskLogger: std::ptr::null_mut(),
            rwConfigParams: Default::default(),
        };
        let mut this = Self { config };
        this.reset(0);
        this
    }

    pub fn reset(&mut self, time: u64) {
        unsafe {
            crate::sys::Reset_mrpFeedback(&mut self.config, time, MODULE_ID);
        }
    }

    pub fn update(&mut self, time: u64) {
        unsafe {
            crate::sys::Update_mrpFeedback(&mut self.config, time, MODULE_ID);
        }
    }
}

#[cfg_attr(feature = "serde", derive(Debug, serde::Serialize, serde::Deserialize))]
pub struct RWArrayConfig {
    pub wheels: Vec<RWConfig>,
}

#[cfg_attr(feature = "serde", derive(Debug, serde::Serialize, serde::Deserialize))]
pub struct RWConfig {
    pub spin_axis: [f64; 3],
    pub inertia: f64,
    pub max_torque: f64,
}

impl From<RWArrayConfig> for RWArrayConfigMsgPayload {
    fn from(val: RWArrayConfig) -> Self {
        let mut gs_matrix_b = [0.0; { 3 * 36 }];
        let mut u_max = [0.0; 36];
        let mut js_list = [0.0; 36];
        let len = val.wheels.len();
        for (i, wheel) in val.wheels.into_iter().enumerate() {
            gs_matrix_b[i * 3..(i + 1) * 3].copy_from_slice(&wheel.spin_axis);
            u_max[i] = wheel.max_torque;
            js_list[i] = wheel.inertia;
        }
        RWArrayConfigMsgPayload {
            GsMatrix_B: gs_matrix_b,
            numRW: len as i32,
            uMax: u_max,
            JsList: js_list,
        }
    }
}

pub struct RWAvailability {
    pub availability: Vec<bool>,
}

impl From<RWAvailability> for RWAvailabilityMsgPayload {
    fn from(val: RWAvailability) -> Self {
        let mut availability = [0; 36]; // 0 = AVAILABLE, 1 = UNAVAILABLE
        for (i, avail) in val.availability.into_iter().enumerate() {
            availability[i] = if avail { 0 } else { 1 };
        }
        RWAvailabilityMsgPayload {
            wheelAvailability: availability,
        }
    }
}

/// MRP steering control module - a wrapper around `mrpSteering`
///
/// See https://hanspeterschaub.info/basilisk/Documentation/fswAlgorithms/attControl/mrpSteering/mrpSteering.html for more info
pub struct MrpPD {
    config: MrpPDConfig,
}

impl MrpPD {
    pub fn new(
        k: f64,
        p: f64,
        known_torque: [f64; 3],
        vehicle_config: BskChannel<VehicleConfigMsgPayload>,
        att_guid: BskChannel<AttGuidMsgPayload>,
        torque_out: BskChannel<CmdTorqueBodyMsgPayload>,
    ) -> Self {
        let mut this = Self {
            config: MrpPDConfig {
                K: k,
                P: p,
                knownTorquePntB_B: known_torque,
                ISCPntB_B: [0.0; 9],
                cmdTorqueOutMsg: torque_out.into(),
                guidInMsg: att_guid.into(),
                vehConfigInMsg: vehicle_config.into(),
                bskLogger: std::ptr::null_mut(),
            },
        };
        this.reset(0);
        this
    }

    pub fn reset(&mut self, time: u64) {
        unsafe { Reset_mrpPD(&mut self.config, time, MODULE_ID) }
    }

    pub fn update(&mut self, time: u64) {
        unsafe { Update_mrpPD(&mut self.config, time, MODULE_ID) }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_mrp_steer_basic() {
        let att_guid = BskChannel::pair();
        let rate_cmd = BskChannel::pair();
        let mut mrp_steer =
            MRPSteering::new(1.0, 1.0, 1.0, true, rate_cmd.clone(), att_guid.clone());
        att_guid.write(
            AttGuidMsgPayload {
                sigma_BR: [1.0, 0.0, 0.0],
                omega_BR_B: [0.0, 0.0, 0.0],
                omega_RN_B: [1.0, 1.0, 1.0],
                domega_RN_B: [1.0, 1.0, 1.0],
            },
            0,
        );
        mrp_steer.update(0);
        let out = rate_cmd.read_msg();
        assert_relative_eq!(
            out.omega_BastR_B.as_ref(),
            [-0.8038134760954126, -0.0, -0.0].as_ref()
        );
        assert_relative_eq!(
            out.omegap_BastR_B.as_ref(),
            [0.14790114643268049, -0.0, -0.0].as_ref()
        );
    }

    #[test]
    fn test_mrp_feedback_basic() {
        let config = MRPFeedbackConfig {
            k: 1.0,
            p: 1.0,
            k_i: 1.0,
            integral_limit: 1.0,
            control_law: MRPFeedbackControlLaw::A,
            prior_time: 0,
            z: [1.0, 1.0, 1.0],
            int_sigma: [0.0, 0.0, 0.0],
            known_torque_pnt_b_b: [0.0, 0.0, 0.0],
            inertia: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        };
        let rw_speed_cmd = BskChannel::pair();
        let rw_array_config = BskChannel::pair();
        rw_array_config.write(
            RWArrayConfig {
                wheels: vec![
                    RWConfig {
                        spin_axis: [1.0, 0.0, 0.0],
                        inertia: 1.0,
                        max_torque: 1.0,
                    },
                    RWConfig {
                        spin_axis: [0.0, 1.0, 0.0],
                        inertia: 1.0,
                        max_torque: 1.0,
                    },
                    RWConfig {
                        spin_axis: [0.0, 0.0, 1.0],
                        inertia: 1.0,
                        max_torque: 1.0,
                    },
                ],
            }
            .into(),
            0,
        );
        let rw_availability_cmd = BskChannel::pair();
        rw_availability_cmd.write(
            RWAvailability {
                availability: vec![true, true, true],
            }
            .into(),
            0,
        );
        let cmd_torque_body = BskChannel::pair();
        let int_feedback = BskChannel::pair();
        let att_guid_cmd = BskChannel::pair();
        att_guid_cmd.write(
            AttGuidMsgPayload {
                sigma_BR: [1.0, 0.0, 0.0],
                omega_BR_B: [0.0, 0.0, 0.0],
                omega_RN_B: [1.0, 1.0, 1.0],
                domega_RN_B: [1.0, 1.0, 1.0],
            },
            0,
        );
        let vehicle_config = BskChannel::pair();
        let mut mrp_feedback = MRPFeedback::new(
            config,
            rw_speed_cmd,
            rw_availability_cmd,
            rw_array_config,
            cmd_torque_body.clone(),
            int_feedback,
            att_guid_cmd,
            vehicle_config,
        );
        mrp_feedback.reset(0);
        mrp_feedback.update(1);
        let cmd_torque = cmd_torque_body.read_msg();
        assert_eq!(
            cmd_torque,
            CmdTorqueBodyMsgPayload {
                torqueRequestBody: [-1.0, 0.0, 0.0]
            }
        );
    }

    #[test]
    fn test_mrp_pd_basic() {
        let att_guid = BskChannel::pair();
        let vehicle_config = BskChannel::pair();
        let cmd_torque = BskChannel::pair();
        let mut mrp_pd = MrpPD::new(
            1.0,
            1.0,
            [0.0, 0.0, 0.0],
            vehicle_config.clone(),
            att_guid.clone(),
            cmd_torque.clone(),
        );
        att_guid.write(
            AttGuidMsgPayload {
                sigma_BR: [1.0, 0.0, 0.0],
                omega_BR_B: [0.0, 0.0, 0.0],
                omega_RN_B: [1.0, 1.0, 1.0],
                domega_RN_B: [1.0, 1.0, 1.0],
            },
            0,
        );
        vehicle_config.write(
            VehicleConfigMsgPayload {
                ISCPntB_B: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                CoM_B: [0.0, 0.0, 0.0],
                massSC: 100.0,
                CurrentADCSState: 0,
            },
            0,
        );
        mrp_pd.reset(0);
        mrp_pd.update(1);
        let cmd_torque = cmd_torque.read_msg();
        assert_eq!(
            cmd_torque,
            CmdTorqueBodyMsgPayload {
                torqueRequestBody: [-0.0, 1.0, 1.0]
            }
        );
    }
}
