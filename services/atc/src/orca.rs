use std::time::Duration;

use atc_entity::vm::{self, Status};
use futures::{select, stream, StreamExt};
use kube::{api::PostParams, core::ObjectMeta, runtime::watcher, runtime::watcher::Event, Api};

use k8s_openapi::api::core::v1::{Container, Pod, PodSpec};
use sea_orm::{ActiveModelTrait, Database, DatabaseConnection, Set, Unchanged};
use tokio::task::JoinHandle;
use tracing::{error, warn};
use uuid::Uuid;

use crate::{config::OrcaConfig, error::Error};

pub struct Orca {
    vm_manager: VmManager,
}

impl Orca {
    pub async fn new(vm_manager: VmManager) -> anyhow::Result<Self> {
        Ok(Self { vm_manager })
    }

    pub fn run(
        mut self,
    ) -> (
        JoinHandle<Result<(), anyhow::Error>>,
        flume::Sender<OrcaMsg>,
    ) {
        let pods: Api<Pod> =
            kube::api::Api::namespaced(self.vm_manager.k8s.clone(), &self.vm_manager.vm_namespace);
        let events = watcher(pods, watcher::Config::default());
        let (tx, rx) = flume::unbounded::<OrcaMsg>();
        let rx = rx.into_stream();
        let mut stream = stream::select(Box::pin(events.map(OrcaMsg::K8sEvent)), Box::pin(rx));
        let handle = tokio::spawn(async move {
            loop {
                let Some(msg) = stream.next().await else {
                    return Ok(());
                };
                match msg {
                    OrcaMsg::K8sEvent(Ok(Event::Applied(pod))) => {
                        if let Err(err) = self.handle_pod_change(&pod).await {
                            warn!(?err, "error handling pot update");
                        }
                    }
                    OrcaMsg::K8sEvent(Ok(Event::Deleted(pod))) => {
                        //TODO(sphw)
                    }
                    OrcaMsg::K8sEvent(Ok(Event::Restarted(pod))) => {
                        //TODO(sphw)
                    }

                    OrcaMsg::K8sEvent(Err(err)) => {
                        error!(?err, "error watching");
                        return Ok(());
                    }
                }
            }
        });
        (handle, tx)
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
        let Ok(id) = Uuid::parse_str(&name) else {
            return Ok(());
        };
        vm::ActiveModel {
            id: Unchanged(id),
            pod_name: Unchanged(name.clone()),
            status: Set(status),
            ..Default::default()
        }
        .update(&self.vm_manager.db)
        .await?;
        Ok(())
    }
}

pub enum OrcaMsg {
    K8sEvent(Result<Event<Pod>, kube::runtime::watcher::Error>),
}

#[derive(Clone)]
pub struct VmManager {
    pub k8s: kube::Client,
    pub db: DatabaseConnection,
    pub vm_namespace: String,
}

impl VmManager {
    pub async fn new(vm_namespace: String, db_url: String) -> anyhow::Result<Self> {
        let k8s = kube::Client::try_default().await?;
        let db = Database::connect(&db_url).await?;
        Ok(Self {
            db,
            k8s,
            vm_namespace,
        })
    }
    pub async fn spawn_vm(&self, id: Uuid, image: String) -> Result<(), Error> {
        let api: Api<Pod> = kube::api::Api::namespaced(self.k8s.clone(), &self.vm_namespace);
        let pod_name = id.to_string();
        match api
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
                            image: Some(image),
                            ..Default::default()
                        }],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )
            .await
        {
            Ok(resp) => resp,
            Err(err) => {
                self.set_vm_status(id, Status::Error).await?;
                return Err(err.into());
            }
        };
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
}
