use config::{ConfigError, Environment, File};
use serde::Deserialize;
use std::net::SocketAddr;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub control_addr: SocketAddr,
    pub sim_addr: SocketAddr,
}

impl Config {
    pub fn new() -> Result<Self, ConfigError> {
        config::Config::builder()
            .add_source(File::with_name("./config.toml").required(false))
            .add_source(File::with_name("/etc/elodin/sim.toml").required(false))
            .add_source(Environment::with_prefix("ELODIN"))
            .build()?
            .try_deserialize()
    }
}
