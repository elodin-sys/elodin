use std::time::Duration;

use crate::{error::Error, events::EntityExt};
use atc_entity::{sandbox, vm};
use chrono::Utc;
use paracosm_types::sandbox::{sandbox_control_client::SandboxControlClient, UpdateCodeReq};
use redis::aio::MultiplexedConnection;
use sea_orm::{ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter};
use tonic::transport::Channel;

pub async fn update_sandbox_code(vm_ip: &str, code: String) -> Result<(), Error> {
    let Ok(ip) = format!("grpc://{}:50051", vm_ip).parse() else {
        return Err(Error::VMBootFailed("vm has invalid ip".to_string()));
    };
    let channel = Channel::builder(ip).connect().await?;
    let mut client = SandboxControlClient::new(channel);
    let res = client
        .update_code(UpdateCodeReq { code })
        .await
        .map_err(|err| Error::VMBootFailed(err.to_string()))?;
    let _res = res.into_inner();
    Ok(())
}

pub async fn garbage_collect(
    db: DatabaseConnection,
    mut redis: MultiplexedConnection,
    timeout: Duration,
) -> anyhow::Result<()> {
    loop {
        tokio::time::sleep(timeout / 10).await;
        let stale_cutoff = Utc::now() - timeout;
        let stale_sandboxes = sandbox::Entity::find()
            .filter(sandbox::Column::LastUsed.lte(stale_cutoff))
            .filter(sandbox::Column::VmId.is_not_null())
            .all(&db)
            .await?;
        for sandbox in stale_sandboxes {
            if let Some(vm_id) = sandbox.vm_id {
                vm::Entity::delete_with_event(vm_id, &db, &mut redis).await?;
            }
        }
    }
}
