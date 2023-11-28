use atc_entity::vm::{self, Status};
use futures::{stream, StreamExt};
use kube::{api::PostParams, core::ObjectMeta, runtime::watcher, runtime::watcher::Event, Api};

use k8s_openapi::api::core::v1::{Container, Pod, PodSpec};
use redis::aio::PubSub;
use sea_orm::{ActiveModelTrait, DatabaseConnection, Set, Unchanged};
use tokio::task::JoinHandle;
use tracing::{error, trace, warn};
use uuid::Uuid;

use crate::{config::OrcaConfig, error::Error, events::DbEvent};

pub struct Orca {
    k8s: kube::Client,
    db: DatabaseConnection,
    vm_namespace: String,
    image_name: String,
}

impl Orca {
    pub async fn new(config: OrcaConfig, db: DatabaseConnection) -> anyhow::Result<Self> {
        let k8s = kube::Client::try_default().await?;
        Ok(Self {
            db,
            k8s,
            vm_namespace: config.vm_namespace,
            image_name: config.image_name,
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
                            warn!(?err, "error handling pot update");
                        }
                    }
                    Ok(OrcaMsg::K8sEvent(Event::Deleted(_pod))) => {
                        // TODO(sphw): add pod manual deletion handler, this should set the VM into some sort of "failed" state
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
        vm::ActiveModel {
            id: Unchanged(id),
            pod_name: Unchanged(name.clone()),
            status: Set(status),
        }
        .update(&self.db)
        .await?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum OrcaMsg {
    DbEvent(DbEvent<vm::Model>),
    K8sEvent(Event<Pod>),
}
