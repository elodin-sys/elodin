use atc_entity::{
    sandbox,
    vm::{self, Status},
};
use futures::{stream, StreamExt};
use k8s_openapi::api::core::v1::{Container, EnvVar, Pod, PodSpec};
use kube::{
    api::{DeleteParams, PostParams},
    core::ObjectMeta,
    runtime::watcher,
    runtime::watcher::Event,
    Api,
};
use redis::aio::{MultiplexedConnection, PubSub};
use sea_orm::{ActiveModelTrait, DatabaseConnection, EntityTrait, Set, Unchanged};
use tokio::task::JoinHandle;
use tracing::{error, trace, warn};
use uuid::Uuid;

use crate::{
    config::OrcaConfig,
    error::Error,
    events::{DbEvent, DbExt, EntityExt},
    sandbox::update_sandbox_code,
};

pub struct Orca {
    k8s: kube::Client,
    db: DatabaseConnection,
    vm_namespace: String,
    image_name: String,
    redis: MultiplexedConnection,
}

impl Orca {
    pub async fn new(
        config: OrcaConfig,
        db: DatabaseConnection,
        redis: MultiplexedConnection,
    ) -> anyhow::Result<Self> {
        let k8s = kube::Client::try_default().await?;
        Ok(Self {
            db,
            k8s,
            vm_namespace: config.vm_namespace,
            image_name: config.image_name,
            redis,
        })
    }

    pub fn run(mut self, mut redis: PubSub) -> JoinHandle<Result<(), anyhow::Error>> {
        let pods: Api<Pod> = kube::api::Api::namespaced(self.k8s.clone(), &self.vm_namespace);
        let events = watcher(pods, watcher::Config::default());
        let handle = tokio::spawn(async move {
            redis.subscribe("vm_events").await?;
            let rx = redis.on_message().map(|msg| {
                postcard::from_bytes(msg.get_payload_bytes())
                    .map(OrcaMsg::DbEvent)
                    .map_err(Error::from)
            });
            let mut stream = stream::select(
                Box::pin(events.map(|e| e.map(OrcaMsg::K8sEvent).map_err(Error::from))),
                Box::pin(rx),
            );

            loop {
                let Some(msg) = stream.next().await else {
                    return Ok(());
                };
                match msg {
                    Err(err) => {
                        error!(?err, "error event");
                    }
                    Ok(OrcaMsg::K8sEvent(Event::Applied(pod))) => {
                        if let Err(err) = self.handle_pod_change(&pod).await {
                            warn!(?err, "error handling pod update");
                        }
                    }
                    Ok(OrcaMsg::K8sEvent(Event::Deleted(pod))) => {
                        if let Err(err) = self.handle_pod_deleted(&pod).await {
                            warn!(?err, "error handling pod deleted");
                        }
                    }
                    Ok(OrcaMsg::K8sEvent(Event::Restarted(_pods))) => {
                        // NOTE(sphw): choosing not to handle stream restarts right now
                    }
                    Ok(OrcaMsg::DbEvent(DbEvent::Insert(vm))) => {
                        trace!(?vm, "vm insert event");
                        if let Err(err) = self.spawn_vm(vm.id).await {
                            error!(?err, "error spawning vm");
                        }
                    }
                    Ok(OrcaMsg::DbEvent(DbEvent::Delete(vm))) => {
                        if let Err(err) = self.handle_vm_deleted(&vm).await {
                            warn!(?err, "error handling vm deleted");
                        }
                    }
                    Ok(OrcaMsg::DbEvent(DbEvent::Update(_vm))) => {}
                }
            }
        });
        handle
    }

    async fn spawn_vm(&mut self, id: Uuid) -> Result<(), Error> {
        let api: Api<Pod> = kube::api::Api::namespaced(self.k8s.clone(), &self.vm_namespace);
        let pod_name = id.to_string();
        if let Err(err) = api
            .create(
                &PostParams::default(),
                &Pod {
                    metadata: ObjectMeta {
                        name: Some(pod_name),
                        ..Default::default()
                    },
                    spec: Some(PodSpec {
                        containers: vec![Container {
                            name: "payload".to_string(),
                            image: Some(self.image_name.clone()),
                            env: Some(vec![
                                EnvVar {
                                    name: "ELODIN_CONTROL_ADDR".to_string(),
                                    value: Some("[::]:50051".to_string()),
                                    ..Default::default()
                                },
                                EnvVar {
                                    name: "ELODIN_SIM_ADDR".to_string(),
                                    value: Some("[::]:3563".to_string()),
                                    ..Default::default()
                                },
                            ]),
                            ..Default::default()
                        }],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )
            .await
        {
            self.set_vm_status(id, Status::Error).await?;
            return Err(err.into());
        }
        Ok(())
    }

    async fn set_vm_status(&self, id: Uuid, status: Status) -> Result<(), Error> {
        vm::ActiveModel {
            id: Unchanged(id),
            status: Set(status),
            ..Default::default()
        }
        .update(&self.db)
        .await?;
        Ok(())
    }

    async fn handle_pod_change(&self, pod: &Pod) -> Result<(), Error> {
        let Some(phase) = pod.status.as_ref().and_then(|s| s.phase.as_ref()) else {
            return Ok(());
        };
        let Some(name) = &pod.metadata.name else {
            return Ok(());
        };
        let status = match phase.as_str() {
            "Pending" => Status::Booting,
            "Running" => Status::Running,
            "Failed" => Status::Error,
            "Unknown" => Status::Error,
            _ => Status::Error,
        };
        let Ok(id) = Uuid::parse_str(name) else {
            return Ok(());
        };
        let pod_ip = pod.status.as_ref().and_then(|s| s.pod_ip.clone());
        let vm = vm::ActiveModel {
            id: Unchanged(id),
            pod_name: Unchanged(name.clone()),
            status: Set(status),
            pod_ip: Set(pod_ip),
            ..Default::default()
        }
        .update(&self.db)
        .await?;
        if let Some(sandbox_id) = vm.sandbox_id {
            self.propagate_sandbox(sandbox_id, &vm).await?;
        }
        Ok(())
    }

    async fn propagate_sandbox(&self, sandbox_id: Uuid, vm: &vm::Model) -> Result<(), Error> {
        let Some(sandbox) = sandbox::Entity::find_by_id(sandbox_id)
            .one(&self.db)
            .await?
        else {
            warn!(?vm, "vm has invalid sandbox id");
            return Ok(());
        };
        let new_sandbox_state = match vm.status {
            Status::Pending => sandbox::Status::Off,
            Status::Booting => sandbox::Status::VmBooting,
            Status::Error => sandbox::Status::Error,
            Status::Running => sandbox::Status::Running,
        };
        sandbox::ActiveModel {
            id: Unchanged(sandbox_id),
            status: Set(new_sandbox_state),
            ..Default::default()
        }
        .update_with_event(&self.db, &mut self.redis.clone())
        .await?;
        match (sandbox.status, new_sandbox_state) {
            (sandbox::Status::Running, sandbox::Status::Running) => {}
            (_, sandbox::Status::Running) => {
                let Some(ref pod_ip) = vm.pod_ip else {
                    warn!("supposed to be unreachable - vm with out pod ip running");
                    return Err(Error::VMBootFailed("pod missing ip".to_string()));
                };
                update_sandbox_code(pod_ip, sandbox.code).await?;
            }
            (_, _) => {}
        }
        Ok(())
    }

    async fn handle_pod_deleted(&mut self, pod: &Pod) -> Result<(), Error> {
        let Some(name) = &pod.metadata.name else {
            return Ok(());
        };
        let Ok(id) = Uuid::parse_str(name) else {
            return Ok(());
        };
        let model = match vm::Entity::delete_with_event(id, &self.db, &mut self.redis).await {
            Ok(model) => model,
            Err(Error::NotFound) | Err(Error::Db(sea_orm::DbErr::RecordNotFound(_))) => {
                return Ok(())
            }
            Err(err) => return Err(err),
        };
        if let Some(sandbox_id) = model.sandbox_id {
            sandbox::ActiveModel {
                id: Unchanged(sandbox_id),
                status: Set(sandbox::Status::Off),
                vm_id: Set(None),
                ..Default::default()
            }
            .update_with_event(&self.db, &mut self.redis)
            .await?;
        }
        Ok(())
    }

    async fn handle_vm_deleted(&mut self, vm: &vm::Model) -> Result<(), Error> {
        let pods: Api<Pod> = kube::api::Api::namespaced(self.k8s.clone(), &self.vm_namespace);
        pods.delete(
            &vm.pod_name,
            &DeleteParams {
                grace_period_seconds: Some(0),
                ..Default::default()
            },
        )
        .await?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum OrcaMsg {
    DbEvent(DbEvent<vm::Model>),
    K8sEvent(Event<Pod>),
}
