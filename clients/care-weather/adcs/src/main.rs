use basilisk::att_determination::SunlineConfig;
use determination::{Determination, MagCal};
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

#[derive(Default, Componentize, Decomponentize, Debug)]
pub struct NavData {
    #[roci(entity_id = 3, component_id = "att_mrp_bn")]
    pub att_mrp_bn: [f64; 3],
    #[roci(entity_id = 3, component_id = "omega_bn_b")]
    pub omega_bn_b: [f64; 3],
    #[roci(entity_id = 0, component_id = "sun_vec_b")]
    pub sun_vec_b: [f64; 3],
}

#[derive(Serialize, Deserialize)]
pub struct Config {
    sunline: SunlineConfig,
    guidance: guidance::GuidanceConfig,
    control: control::ControlConfig,
    mcu: mcu::McuConfig,
    mekf: Option<determination::MEKFConfig>,
    #[serde(default)]
    mag_cal: MagCal,
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
        mcu,
        mekf,
        mag_cal,
    } = Config::parse()?;
    let det = Determination::new(sunline, mag_cal, mekf);
    let _guidance = guidance::Guidance::new(guidance.sigma_r0r);
    let _control = control::Control::new(control);
    let (tx, _) = tokio::tcp_connect::<Hz<100>>(
        "127.0.0.1:2240".parse().unwrap(),
        &[],
        sim_adapter::TxWorld::metadata(),
    );
    let mut mcu_driver = mcu::McuDriver::new(mcu.path, mcu.baud_rate)?;
    let adcs_format = mcu::AdcsFormat::IncludeMag
        | mcu::AdcsFormat::IncludeGyro
        | mcu::AdcsFormat::IncludeAccel
        | mcu::AdcsFormat::IncludeCss;
    mcu_driver.print_info().unwrap();
    mcu_driver.init_adcs(100, 20, 10, adcs_format)?;
    if let Err(err) = mcu_driver.init_gps(1000, 20, 10) {
        eprintln!("failed to initialize GPS: {}", err);
    }
    let sim_adapter = sim_adapter::SimAdapter;
    os_sleep_driver(mcu_driver.pipe(det).pipe(sim_adapter).pipe(tx)).run();
    Ok(())
}
