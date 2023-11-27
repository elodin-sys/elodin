use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::{ffi::OsStr, fs, net::SocketAddr, path::Path};
use tracing::info;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    pub api: Option<ApiConfig>,
    pub orca: Option<OrcaConfig>,
    pub database_url: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ApiConfig {
    pub address: SocketAddr,
    pub auth0: Auth0Config,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Auth0Config {
    pub domain: String,
    pub client_id: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OrcaConfig {
    pub vm_namespace: String,
}

impl Config {
    pub fn parse() -> anyhow::Result<Self> {
        let Some(path) = Path::from_exists("./config.toml")
            .or_else(|| Path::from_exists("/etc/elodin/atc.toml"))
        else {
            return Err(anyhow!("config not found"));
        };
        info!(config.path = ?path, "found config");
        let config = fs::read_to_string(path)?;
        toml::from_str(&config).map_err(anyhow::Error::from)
    }
}

trait PathExt {
    fn from_exists(s: &(impl AsRef<OsStr> + ?Sized)) -> Option<&Path>;
}

impl PathExt for Path {
    fn from_exists(s: &(impl AsRef<OsStr> + ?Sized)) -> Option<&Path> {
        let p = Path::new(s);
        p.exists().then_some(p)
    }
}
