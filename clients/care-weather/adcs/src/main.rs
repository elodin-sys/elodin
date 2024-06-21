use basilisk::att_determination::SunlineConfig;
use conduit::Query;
use determination::Determination;
use guidance::{Guidance, GuidanceConfig};
use roci::{
    combinators::PipeExt,
    drivers::{os_sleep_driver, Driver, Hz},
    tokio, Componentize, Decomponentize, System,
};
use serde::{Deserialize, Serialize};
use sim_adapter::SimAdapter;

mod control;
mod determination;
mod guidance;
pub mod mcu;
mod sim_adapter;

#[derive(Default, Componentize, Decomponentize, Debug)]
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
    pub fn parse() -> anyhow::Result<Self> {
        let config_paths = [
            std::env::var("CW_ADCS_CONFIG")
                .unwrap_or_else(|_| "/etc/care-weather/adcs.toml".to_string()),
            "./config.toml".to_string(),
        ];
        for path in config_paths {
            let Ok(config) = std::fs::read_to_string(path) else {
                continue;
            };
            return toml::from_str(&config).map_err(anyhow::Error::from);
        }
        panic!("No config file found");
    }
}

fn main() -> anyhow::Result<()> {
    let Config {
        sunline,
        guidance,
        control,
    } = Config::parse()?;
    let det = Determination::new(sunline);
    let guidance = Guidance::new(guidance.sigma_r0r);
    let control = control::Control::new(control);
    let (tx, rx) = tokio::tcp_listen::<Hz<100>>(
        "127.0.0.1:2241".parse()?,
        &[],
        <Determination as System>::World::metadata(),
    );
    let (sim_tx, sim_rx) = tokio::tcp_connect::<Hz<100>>(
        "127.0.0.1:2240".parse().unwrap(),
        &[
            Query::with_id("css_value"),
            Query::with_id("rw_speed"),
            Query::with_id("world_pos"),
            Query::with_id("world_vel"),
        ],
        control::World::metadata(),
    );

    os_sleep_driver(
        tx.pipe(sim_rx)
            .pipe(SimAdapter)
            .pipe(det)
            .pipe(guidance)
            .pipe(control)
            .pipe(sim_tx)
            .pipe(rx),
    )
    .run();
    Ok(())
}
