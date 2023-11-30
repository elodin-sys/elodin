use atc_entity::{sandbox, user, vm};
use compact_str::CompactString;
use redis::AsyncCommands;
use sea_orm::{
    ActiveModelBehavior, ActiveModelTrait, ConnectionTrait, EntityTrait, IntoActiveModel,
};
use serde::{Deserialize, Serialize};
use tonic::async_trait;

use crate::error::Error;

pub trait EventModel {
    fn topic_name(&self) -> CompactString;
}

#[async_trait]
pub trait DbExt: ActiveModelTrait {
    async fn update_with_event(
        self,
        db: &impl ConnectionTrait,
        redis: &mut redis::aio::MultiplexedConnection,
    ) -> Result<(), Error>;
    async fn insert_with_event(
        self,
        db: &impl ConnectionTrait,
        redis: &mut redis::aio::MultiplexedConnection,
    ) -> Result<(), Error>;
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
    ) -> Result<(), Error> {
        let model = self.update(db).await?;
        let topic_name = model.topic_name();
        let event = DbEvent::Update(model);
        let buf = postcard::to_allocvec(&event)?;
        redis.publish::<&str, &[u8], _>(&topic_name, &buf).await?;
        Ok(())
    }

    async fn insert_with_event(
        self,
        db: &impl ConnectionTrait,
        redis: &mut redis::aio::MultiplexedConnection,
    ) -> Result<(), Error> {
        let model = self.insert(db).await?;
        let topic_name = model.topic_name();
        let event = DbEvent::Insert(model);
        let buf = postcard::to_allocvec(&event)?;
        redis.publish::<&str, &[u8], _>(&topic_name, &buf).await?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum DbEvent<M> {
    Insert(M),
    Update(M),
}

impl EventModel for vm::Model {
    fn topic_name(&self) -> CompactString {
        "vm_events".into()
    }
}

impl EventModel for user::Model {
    fn topic_name(&self) -> CompactString {
        "user_events".into()
    }
}

impl EventModel for sandbox::Model {
    fn topic_name(&self) -> CompactString {
        "sandbox_events".into()
    }
}
