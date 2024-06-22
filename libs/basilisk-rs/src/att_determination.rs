use crate::{
    channel::BskChannel,
    sys::{
        sunlineEKFConfig, CSSArraySensorMsgPayload, CSSConfigMsgPayload, CSSUnitConfigMsgPayload,
        NavAttMsgPayload, SunlineFilterMsgPayload,
    },
};

pub const MODULE_ID: i64 = 0x224D;

pub struct SunlineEKF {
    config: sunlineEKFConfig,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct CSSConfig {
    normal_b: [f64; 3],
    coefficent: f64,
}

impl From<CSSConfig> for CSSUnitConfigMsgPayload {
    fn from(val: CSSConfig) -> Self {
        CSSUnitConfigMsgPayload {
            nHat_B: val.normal_b,
            CBias: val.coefficent,
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SunlineConfig {
    css: Vec<CSSConfig>,
    q_obs_val: f64,
    q_proc_val: f64,
    covar: Vec<f64>,
    sensor_use_threshold: f64,
    ekf_switch: f64,
}

impl SunlineEKF {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nav_state_out: BskChannel<NavAttMsgPayload>,
        filter_data_out: BskChannel<SunlineFilterMsgPayload>,
        css_input: BskChannel<CSSArraySensorMsgPayload>,
        config: SunlineConfig,
    ) -> Self {
        let css_config = BskChannel::default();
        let SunlineConfig {
            css,
            q_obs_val,
            q_proc_val,
            covar,
            sensor_use_threshold,
            ekf_switch,
        } = config;
        let post_fit_residuals = [0.1; 32];
        let mut css_vals = [CSSUnitConfigMsgPayload::default(); 32];
        let css_len = css.len();
        for (i, css) in css.into_iter().enumerate() {
            css_vals[i] = css.into();
        }
        css_config.write(
            CSSConfigMsgPayload {
                nCSS: css_len as u32,
                cssVals: css_vals,
            },
            0,
        );
        let covar = covar.as_slice().try_into().expect("covar array must be 36");
        let mut this = Self {
            config: sunlineEKFConfig {
                navStateOutMsg: nav_state_out.into(),
                filtDataOutMsg: filter_data_out.into(),
                cssDataInMsg: css_input.into(),
                cssConfigInMsg: css_config.into(),
                qObsVal: q_obs_val,
                qProcVal: q_proc_val,
                dt: 0.1,
                timeTag: f64::NAN,
                state: [0.0, 1.0, 0.0, 0.0, 0.01, 0.0],
                x: [0.1; 6],
                xBar: [0.1; 6],
                covarBar: [0.1; 36],
                covar,
                stateTransition: [0.1; 36],
                kalmanGain: [0.1; 192],
                dynMat: [0.1; 36],
                measMat: [0.1; 192],
                obs: [0.1; 32],
                yMeas: [0.1; 32],
                procNoise: [0.1; 9],
                measNoise: [0.1; 1024],
                postFits: post_fit_residuals,
                cssNHat_B: [0.1; 96],
                CBias: [0.1; 32],
                numStates: 1,
                numObs: 1,
                numActiveCss: 3,
                numCSSTotal: 3,
                sensorUseThresh: sensor_use_threshold,
                eKFSwitch: ekf_switch,
                outputSunline: NavAttMsgPayload::default(),
                cssSensorInBuffer: CSSArraySensorMsgPayload::default(),
                bskLogger: std::ptr::null_mut(),
            },
        };
        this.reset();
        this
    }

    pub fn reset(&mut self) {
        unsafe { crate::sys::Reset_sunlineEKF(&mut self.config, 0, MODULE_ID) }
    }

    pub fn update(&mut self, time: u64) {
        unsafe { crate::sys::Update_sunlineEKF(&mut self.config, time, MODULE_ID) }
    }
}
