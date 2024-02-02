use atc_entity::{sandbox, user, vm};
use compact_str::CompactString;
use futures::StreamExt;
use redis::{aio::PubSub, AsyncCommands};
use sea_orm::{
    ActiveModelBehavior, ActiveModelTrait, ConnectionTrait, EntityTrait, IntoActiveModel,
    PrimaryKeyTrait,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
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

#[async_trait]
pub trait EntityExt: EntityTrait {
    async fn delete_with_event(
        id: <Self::PrimaryKey as PrimaryKeyTrait>::ValueType,
        db: &impl ConnectionTrait,
        redis: &mut redis::aio::MultiplexedConnection,
    ) -> Result<Self::Model, Error>;
}

#[async_trait]
impl<M: EntityTrait> EntityExt for M
where
    <Self::PrimaryKey as PrimaryKeyTrait>::ValueType: Clone + Send + Sync,
    Self::Model: Serialize + EventModel,
{
    async fn delete_with_event(
        id: <Self::PrimaryKey as PrimaryKeyTrait>::ValueType,
        db: &impl ConnectionTrait,
        redis: &mut redis::aio::MultiplexedConnection,
    ) -> Result<Self::Model, Error> {
        let topic_name = Self::Model::topic_name();
        let Some(model) = Self::find_by_id(id.clone()).one(db).await? else {
            return Err(Error::NotFound);
        };
        Self::delete_by_id(id).exec(db).await?;
        let event = DbEvent::Delete(model);
        let buf = postcard::to_allocvec(&event)?;
        redis.publish::<&str, &[u8], _>(&topic_name, &buf).await?;
        let DbEvent::Delete(model) = event else {
            unreachable!()
        };
        Ok(model)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum DbEvent<M> {
    Insert(M),
    Update(M),
    Delete(M),
}

impl<M> DbEvent<M> {
    pub fn into_model(self) -> M {
        match self {
            DbEvent::Insert(m) => m,
            DbEvent::Update(m) => m,
            DbEvent::Delete(m) => m,
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
    pub async fn run(
        self,
        mut redis: PubSub,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<()> {
        let cancel_on_drop = cancel_token.clone().drop_guard();
        redis.subscribe(M::topic_name().as_str()).await?;
        let mut stream = redis.on_message();
        loop {
            let msg = tokio::select! {
                _ = cancel_token.cancelled() => break,
                Some(msg) = stream.next() => msg,
                else => break,
            };
            let msg = postcard::from_bytes(msg.get_payload_bytes())?;
            let _ = self.tx.send(msg);
        }
        drop(cancel_on_drop);
        Ok(())
    }
}
