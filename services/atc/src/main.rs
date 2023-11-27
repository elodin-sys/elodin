use anyhow::anyhow;
use api::Api;
use config::Config;
use futures::future;
use tracing::info;

use crate::orca::{Orca, VmManager};

mod actor;
mod api;
mod config;
mod error;
mod orca;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let config = Config::parse()?;
    info!(?config, "config");
    let mut services = vec![];
    let vm_manager = if let Some(orca_config) = config.orca {
        let vm_manager =
            VmManager::new(orca_config.vm_namespace, config.database_url.clone()).await?;
        let orca = Orca::new(vm_manager.clone()).await?;
        let (handle, _) = orca.run();
        services.push(handle);
        Some(vm_manager)
    } else {
        None
    };
    if let Some(api_config) = config.api {
        let Some(vm_manager) = vm_manager else {
            return Err(anyhow!("orca config required for api"))?;
        };
        let api = Api::new(api_config, config.database_url.clone(), vm_manager).await?;
        services.push(tokio::spawn(api.run()));
    }

    let (res, _, _) = future::select_all(services.into_iter()).await;
    res?
}
