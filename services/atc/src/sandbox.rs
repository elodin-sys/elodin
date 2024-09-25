use std::time::Duration;

use crate::error::Error;
use atc_entity::{events::EntityExt, sandbox, vm};
use chrono::Utc;
use elodin_types::sandbox::{
    sandbox_control_client::SandboxControlClient, UpdateCodeReq, UpdateCodeResp,
};
use fred::prelude::*;
use sea_orm::{ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter};
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

pub async fn update_sandbox_code(vm_ip: &str, code: String) -> Result<UpdateCodeResp, Error> {
    let Ok(ip) = format!("grpc://{}:50051", vm_ip).parse() else {
        return Err(Error::VMBootFailed("vm has invalid ip".to_string()));
    };
    let channel = Channel::builder(ip).connect().await?;
    let mut client = SandboxControlClient::new(channel);
    let res = client
        .update_code(UpdateCodeReq { code })
        .await
        .map_err(|err| Error::VMBootFailed(err.to_string()))?;
    Ok(res.into_inner())
}

pub async fn garbage_collect(
    db: DatabaseConnection,
    redis: RedisClient,
    timeout: Duration,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    let cancel_on_drop = cancel_token.clone().drop_guard();
    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => break,
            _ = tokio::time::sleep(timeout / 10) => {},
        };
        let stale_cutoff = Utc::now() - timeout;
        let stale_sandboxes = sandbox::Entity::find()
            .filter(sandbox::Column::LastUsed.lte(stale_cutoff))
            .filter(sandbox::Column::VmId.is_not_null())
            .all(&db)
            .await?;
        for sandbox in stale_sandboxes {
            if let Some(vm_id) = sandbox.vm_id {
                vm::Entity::delete_with_event(vm_id, &db, &redis).await?;
            }
        }
    }
    drop(cancel_on_drop);
    tracing::debug!("garbage_collect - done");
    Ok(())
}
