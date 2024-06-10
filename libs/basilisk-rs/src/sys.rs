#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(rustdoc::broken_intra_doc_links)]

use crate::channel::BskChannel;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

macro_rules! impl_basilisk_msg_c_drop {
    ($msg_name:tt, $payload_name:tt) => {
        unsafe impl Send for $msg_name {}
        unsafe impl Send for $payload_name {}
        impl Drop for $msg_name {
            fn drop(&mut self) {
                let _ = unsafe { BskChannel::<$payload_name>::from_raw(self.payloadPointer as _) };
                self.payloadPointer = std::ptr::null_mut();
                self.headerPointer = std::ptr::null_mut();
            }
        }
    };
}

impl_basilisk_msg_c_drop!(AttGuidMsg_C, AttGuidMsgPayload);
impl_basilisk_msg_c_drop!(RateCmdMsg_C, RateCmdMsgPayload);
impl_basilisk_msg_c_drop!(RWSpeedMsg_C, RWSpeedMsgPayload);
impl_basilisk_msg_c_drop!(RWAvailabilityMsg_C, RWAvailabilityMsgPayload);
impl_basilisk_msg_c_drop!(RWArrayConfigMsg_C, RWArrayConfigMsgPayload);
impl_basilisk_msg_c_drop!(CmdTorqueBodyMsg_C, CmdTorqueBodyMsgPayload);
impl_basilisk_msg_c_drop!(VehicleConfigMsg_C, VehicleConfigMsgPayload);
impl_basilisk_msg_c_drop!(NavAttMsg_C, NavAttMsgPayload);
impl_basilisk_msg_c_drop!(SunlineFilterMsg_C, SunlineFilterMsgPayload);
impl_basilisk_msg_c_drop!(CSSArraySensorMsg_C, CSSArraySensorMsgPayload);
impl_basilisk_msg_c_drop!(CSSConfigMsg_C, CSSConfigMsgPayload);
impl_basilisk_msg_c_drop!(AttRefMsg_C, AttRefMsgPayload);
impl_basilisk_msg_c_drop!(NavTransMsg_C, NavTransMsgPayload);
impl_basilisk_msg_c_drop!(EphemerisMsg_C, EphemerisMsgPayload);
impl_basilisk_msg_c_drop!(ArrayMotorTorqueMsg_C, ArrayMotorTorqueMsgPayload);
impl_basilisk_msg_c_drop!(ArrayMotorVoltageMsg_C, ArrayMotorVoltageMsgPayload);

unsafe impl Send for attTrackingErrorConfig {}
unsafe impl Send for hillPointConfig {}
unsafe impl Send for sunlineEKFConfig {}
unsafe impl Send for mrpFeedbackConfig {}
unsafe impl Send for rwMotorTorqueConfig {}
unsafe impl Send for rwMotorVoltageConfig {}
unsafe impl Send for MrpPDConfig {}
