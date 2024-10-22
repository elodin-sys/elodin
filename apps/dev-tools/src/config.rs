use config::{ConfigError, Environment};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    pub auth0_token: Option<String>,
    pub stripe_token: Option<String>,
}

impl Config {
    pub fn new() -> Result<Self, ConfigError> {
        config::Config::builder()
            .add_source(Environment::with_prefix("ELODIN"))
            .build()?
            .try_deserialize()
    }
}
