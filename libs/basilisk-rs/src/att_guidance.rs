use crate::{
    channel::BskChannel,
    sys::{
        AttGuidMsgPayload, AttRefMsgPayload, EphemerisMsgPayload, NavAttMsgPayload,
        NavTransMsgPayload, attTrackingErrorConfig, hillPointConfig,
    },
};

pub const MODULE_ID: i64 = 0x22467;

pub struct AttTrackingError {
    config: attTrackingErrorConfig,
}

impl AttTrackingError {
    pub fn new(
        sigma_r0r: [f64; 3],
        att_guid_out: BskChannel<AttGuidMsgPayload>,
        nav_att_in: BskChannel<NavAttMsgPayload>,
        att_ref_in: BskChannel<AttRefMsgPayload>,
    ) -> Self {
        let config = attTrackingErrorConfig {
            sigma_R0R: sigma_r0r,
            attGuidOutMsg: att_guid_out.into(),
            attNavInMsg: nav_att_in.into(),
            attRefInMsg: att_ref_in.into(),
            bskLogger: std::ptr::null_mut(),
        };

        let mut this = Self { config };
        this.reset();
        this
    }

    pub fn reset(&mut self) {
        unsafe { crate::sys::Reset_attTrackingError(&mut self.config, 0, MODULE_ID) }
    }

    pub fn update(&mut self, time: u64) {
        unsafe { crate::sys::Update_attTrackingError(&mut self.config, time, MODULE_ID) }
    }
}

pub struct HillPoint {
    config: hillPointConfig,
}

impl HillPoint {
    pub fn new(
        att_ref_out: BskChannel<AttRefMsgPayload>,
        nav_trans_in: BskChannel<NavTransMsgPayload>,
        celestial_body_in: BskChannel<EphemerisMsgPayload>,
    ) -> Self {
        let config = hillPointConfig {
            attRefOutMsg: att_ref_out.into(),
            transNavInMsg: nav_trans_in.into(),
            celBodyInMsg: celestial_body_in.into(),
            bskLogger: std::ptr::null_mut(),
            planetMsgIsLinked: 0,
        };

        let mut this = Self { config };
        this.reset();
        this
    }

    pub fn reset(&mut self) {
        unsafe { crate::sys::Reset_hillPoint(&mut self.config, 0, MODULE_ID) }
    }

    pub fn update(&mut self, time: u64) {
        unsafe { crate::sys::Update_hillPoint(&mut self.config, time, MODULE_ID) }
    }
}
