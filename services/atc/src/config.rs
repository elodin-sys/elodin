use config::{ConfigError, Environment, File};
use redact::serde::redact_secret;
use redact::Secret;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::{net::SocketAddr, time::Duration};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    pub api: Option<ApiConfig>,
    pub orca: Option<OrcaConfig>,
    pub garbage_collect: Option<GarbageCollect>,
    #[serde(default)]
    pub monte_carlo: MonteCarloConfig,
    pub database_url: String,
    pub redis_url: String,
    pub migrate: bool,
    pub pod_name: String,
    pub env: ElodinEnvironment,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum ElodinEnvironment {
    Local,
    DevBranch,
    Dev,
    Prod,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ApiConfig {
    pub address: SocketAddr,
    pub auth0: Auth0Config,
    pub base_url: String,
    #[serde(serialize_with = "redact_secret")]
    pub stripe_secret_key: Secret<String>,
    #[serde(serialize_with = "redact_secret")]
    pub stripe_webhook_secret: Secret<Option<String>>,
    pub stripe_plans: StripePlansConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StripePlansConfig {
    pub commercial: StripePlanConfig,
    pub non_commercial: StripePlanConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StripePlanConfig {
    pub monte_carlo_price: String,
    pub subscription_price: String,
    pub trial_length: u64,
    pub monte_carlo_credit: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Auth0Config {
    pub domain: String,
    pub client_id: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OrcaConfig {
    pub vm_namespace: String,
    pub image_name: String,
    pub runtime_class: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct MonteCarloConfig {
    pub azure_account_name: String,
    pub sim_artifacts_bucket_name: String,
    pub sim_results_bucket_name: String,
}

#[serde_as]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GarbageCollect {
    pub enabled: bool,
    #[serde(default = "default_gc_timeout")]
    #[serde_as(as = "serde_with::DurationSecondsWithFrac<f64>")]
    pub timeout: Duration,
}

fn default_gc_timeout() -> Duration {
    Duration::from_secs(5 * 60)
}

impl Config {
    pub fn new() -> Result<Self, ConfigError> {
        config::Config::builder()
            .add_source(File::with_name("./config.toml").required(false))
            .add_source(File::with_name("/etc/elodin/atc.toml").required(false))
            .add_source(Environment::with_prefix("ELODIN"))
            .build()?
            .try_deserialize()
    }
}
