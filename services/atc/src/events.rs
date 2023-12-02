use atc_entity::{sandbox, user, vm};
use compact_str::CompactString;
use futures::StreamExt;
use redis::{aio::PubSub, AsyncCommands};
use sea_orm::{
    ActiveModelBehavior, ActiveModelTrait, ConnectionTrait, EntityTrait, IntoActiveModel,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::{sync::broadcast, task::JoinHandle};
use tonic::async_trait;

use crate::error::Error;

pub trait EventModel {
    fn topic_name() -> CompactString;
}

#[async_trait]
pub trait DbExt: ActiveModelTrait {
    async fn update_with_event(
        self,
        db: &impl ConnectionTrait,
        redis: &mut redis::aio::MultiplexedConnection,
    ) -> Result<<Self::Entity as EntityTrait>::Model, Error>;
    async fn insert_with_event(
        self,
        db: &impl ConnectionTrait,
        redis: &mut redis::aio::MultiplexedConnection,
    ) -> Result<<Self::Entity as EntityTrait>::Model, Error>;
}

#[async_trait]
impl<A: ActiveModelTrait + ActiveModelBehavior + Send + Sync> DbExt for A
where
    <Self::Entity as EntityTrait>::Model: IntoActiveModel<Self> + EventModel + Serialize,
{
    async fn update_with_event(
        self,
        db: &impl ConnectionTrait,
        redis: &mut redis::aio::MultiplexedConnection,
    ) -> Result<<Self::Entity as EntityTrait>::Model, Error> {
        let model = self.update(db).await?;
        let topic_name = <Self::Entity as EntityTrait>::Model::topic_name();
        let event = DbEvent::Update(model);
        let buf = postcard::to_allocvec(&event)?;
        redis.publish::<&str, &[u8], _>(&topic_name, &buf).await?;
        let DbEvent::Update(model) = event else {
            unreachable!()
        };

        Ok(model)
    }

    async fn insert_with_event(
        self,
        db: &impl ConnectionTrait,
        redis: &mut redis::aio::MultiplexedConnection,
    ) -> Result<<Self::Entity as EntityTrait>::Model, Error> {
        let model = self.insert(db).await?;
        let topic_name = <Self::Entity as EntityTrait>::Model::topic_name();
        let event = DbEvent::Insert(model);
        let buf = postcard::to_allocvec(&event)?;
        redis.publish::<&str, &[u8], _>(&topic_name, &buf).await?;
        let DbEvent::Insert(model) = event else {
            unreachable!()
        };
        Ok(model)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum DbEvent<M> {
    Insert(M),
    Update(M),
}

impl<M> DbEvent<M> {
    pub fn into_model(self) -> M {
        match self {
            DbEvent::Insert(m) => m,
            DbEvent::Update(m) => m,
        }
    }
}

impl EventModel for vm::Model {
    fn topic_name() -> CompactString {
        "vm_events".into()
    }
}

impl EventModel for user::Model {
    fn topic_name() -> CompactString {
        "user_events".into()
    }
}

impl EventModel for sandbox::Model {
    fn topic_name() -> CompactString {
        "sandbox_events".into()
    }
}

pub struct EventMonitor<M> {
    tx: broadcast::Sender<DbEvent<M>>,
}

impl<M: EventModel + DeserializeOwned + Send + 'static + Clone> EventMonitor<M> {
    pub fn pair() -> (Self, broadcast::Receiver<DbEvent<M>>) {
        let (tx, rx) = broadcast::channel(128);
        (EventMonitor { tx }, rx)
    }
    pub fn run(self, mut redis: PubSub) -> JoinHandle<anyhow::Result<()>> {
        tokio::spawn(async move {
            redis.subscribe(M::topic_name().as_str()).await?;
            let mut stream = redis.on_message();
            while let Some(msg) = stream.next().await {
                let msg = postcard::from_bytes(msg.get_payload_bytes())?;
                let _ = self.tx.send(msg);
            }
            Ok(())
        })
    }
}
