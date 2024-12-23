use config::{ConfigError, Environment, File};
use serde::Deserialize;
use std::net::SocketAddr;
use tonic::transport::Uri;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub sandbox: Option<SandboxConfig>,
    pub monte_carlo: Option<MonteCarloConfig>,
}

#[derive(Debug, Deserialize)]
pub struct SandboxConfig {
    pub control_addr: SocketAddr,
    pub sim_addr: SocketAddr,
    #[serde(with = "http_serde::uri", default = "default_vm_addr")]
    pub builder_addr: Uri,
}

#[derive(Debug, Deserialize)]
pub struct MonteCarloConfig {
    pub redis_url: String,
    pub database_url: String,
    pub pod_name: String,
    pub azure_account_name: String,
    pub sim_artifacts_bucket_name: String,
    pub sim_results_bucket_name: String,
    #[serde(with = "http_serde::uri", default = "default_vm_addr")]
    pub tester_addr: tonic::transport::Uri,
}

fn default_vm_addr() -> Uri {
    let addr = format!("vsock://{}:50051", cid());
    addr.parse().unwrap()
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

fn cksum(input: &[u8]) -> u32 {
    const CKSUM: crc::Crc<u32> = crc::Crc::<u32>::new(&crc::CRC_32_CKSUM);
    let mut digest = CKSUM.digest();
    digest.update(input);
    let mut len = input.len();
    while len > 0 {
        let len_octet = len as u8;
        digest.update(&[len_octet]);
        len >>= 8;
    }
    digest.finalize()
}

fn cid() -> u32 {
    let hostname = gethostname::gethostname();
    cksum(hostname.as_encoded_bytes())
}
