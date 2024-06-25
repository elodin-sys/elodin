use roci::{
    combinators::PipeExt,
    drivers::{os_sleep_driver, Driver, Hz},
    tokio, Componentize, Decomponentize,
};
use serde::{Deserialize, Serialize};

mod control;
mod determination;
mod guidance;
pub mod mcu;
mod sim_adapter;

const HZ: usize = 10;

#[derive(Default, Componentize, Decomponentize, Debug)]
pub struct NavData {
    #[roci(entity_id = 3, component_id = "att_mrp_bn")]
    pub att_mrp_bn: [f64; 3],
    #[roci(entity_id = 3, component_id = "omega_bn_b")]
    pub omega_bn_b: [f64; 3],
    #[roci(entity_id = 0, component_id = "sun_vec_b")]
    pub sun_vec_b: [f64; 3],
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    determination: determination::DeterminationConfig,
    guidance: guidance::GuidanceConfig,
    control: control::ControlConfig,
    mcu: mcu::McuConfig,
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
    let config = Config::parse()?;
    let Config {
        determination,
        guidance,
        control,
        mcu,
    } = config;
    let det = determination::Determination::new(determination);
    let _guidance = guidance::Guidance::<HZ>::new(guidance.sigma_r0r);
    let _control = control::Control::<HZ>::new(control);
    let (tx, _) = tokio::tcp_connect::<Hz<HZ>>(
        "127.0.0.1:2240".parse().unwrap(),
        &[],
        sim_adapter::TxWorld::metadata(),
    );
    let mut mcu_driver = mcu::McuDriver::new(mcu)?;
    mcu_driver.print_info().unwrap();
    let sim_adapter = sim_adapter::SimAdapter;
    os_sleep_driver(mcu_driver.pipe(det).pipe(sim_adapter).pipe(tx)).run();
    Ok(())
}
