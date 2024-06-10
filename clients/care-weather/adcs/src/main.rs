use std::time::Duration;

use basilisk::att_determination::SunlineConfig;
use conduit::Query;
use determination::Determination;
use guidance::{Guidance, GuidanceConfig};
use roci::{Componentize, Decomponentize};
use serde::{Deserialize, Serialize};

mod control;
mod determination;
mod guidance;

#[derive(Default, Componentize, Decomponentize)]
pub struct NavData {
    #[roci(entity_id = 3, component_id = "att_mrp_bn")]
    pub att_mrp_bn: [f64; 3],
    #[roci(entity_id = 3, component_id = "omega_bn_b")]
    pub omega_bn_b: [f64; 3],
    #[roci(entity_id = 3, component_id = "sun_vec_b")]
    pub sun_vec_b: [f64; 3],
}

#[derive(Serialize, Deserialize)]
pub struct Config {
    sunline: SunlineConfig,
    guidance: GuidanceConfig,
    control: control::ControlConfig,
}

impl Config {
    pub fn parse() -> Self {
        let config_paths = [
            std::env::var("CW_ADCS_CONFIG")
                .unwrap_or_else(|_| "/etc/care-weather/adcs.toml".to_string()),
            "./config.toml".to_string(),
        ];
        for path in config_paths {
            let Ok(config) = std::fs::read_to_string(path) else {
                continue;
            };
            return toml::from_str(&config).unwrap();
        }
        panic!("No config file found");
    }
}

fn main() {
    let Config {
        sunline,
        guidance,
        control,
    } = Config::parse();
    let det = Determination::new(sunline);
    let guidance = Guidance::new(guidance.sigma_r0r);
    let control = control::Control::new(control);
    let (det_handle, det_channel) = roci::tokio::builder(
        det,
        Duration::from_secs_f64(1.0 / 100.0),
        "127.0.0.1:2241".parse().unwrap(),
    )
    .run();
    let (guidance_handle, guidance_channel) = roci::tokio::builder(
        guidance,
        Duration::from_secs_f64(1.0 / 100.0),
        "127.0.0.1:2242".parse().unwrap(),
    )
    .subscribe(Query::with_id("att_mrp_bn"), det_channel.clone())
    .subscribe(Query::with_id("omega_bn_b"), det_channel.clone())
    .subscribe(Query::with_id("sun_vec_b"), det_channel.clone())
    .run();
    roci::tokio::builder(
        control,
        Duration::from_secs_f64(1.0 / 100.0),
        "127.0.0.1:2243".parse().unwrap(),
    )
    .subscribe(Query::with_id("att_err_mrp"), guidance_channel.clone())
    .subscribe(Query::with_id("omega_err_br_p"), guidance_channel.clone())
    .subscribe(Query::with_id("omega_err_rn_b"), guidance_channel.clone())
    .subscribe(Query::with_id("domega_rn_b"), guidance_channel);
    det_handle.join().unwrap();
    guidance_handle.join().unwrap();
}
