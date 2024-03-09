use futures_util::StreamExt;
use redis::{aio::PubSub, AsyncCommands};
use sea_orm::prelude::*;
use sea_orm::IntoActiveModel;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::sync::broadcast;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("db error: {0}")]
    Db(#[from] sea_orm::DbErr),
    #[error("not found")]
    NotFound,
    #[error("postcard: {0}")]
    Postcard(#[from] postcard::Error),
    #[error("redis: {0}")]
    Redis(#[from] redis::RedisError),
}

pub trait EventModel {
    fn topic_name() -> String;
}

impl<E: ModelTrait + Serialize + DeserializeOwned> EventModel for E {
    fn topic_name() -> String {
        // TODO: memoize this if necessary
        let entity = <Self as ModelTrait>::Entity::default();
        let table_name = entity.table_name();
        format!("{table_name}_events")
    }
}

#[async_trait::async_trait]
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

#[async_trait::async_trait]
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

#[async_trait::async_trait]
pub trait EntityExt: EntityTrait {
    async fn delete_with_event(
        id: <Self::PrimaryKey as PrimaryKeyTrait>::ValueType,
        db: &impl ConnectionTrait,
        redis: &mut redis::aio::MultiplexedConnection,
    ) -> Result<Self::Model, Error>;
}

#[async_trait::async_trait]
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

pub async fn subscribe<M: EventModel + DeserializeOwned + Send + 'static + Clone>(
    mut redis: PubSub,
) -> Result<broadcast::Receiver<DbEvent<M>>, Error> {
    let (tx, rx) = broadcast::channel(128);
    redis.subscribe(M::topic_name().as_str()).await?;
    let mut stream = redis.into_on_message();
    tokio::spawn(async move {
        while let Some(msg) = stream.next().await {
            let msg = postcard::from_bytes(msg.get_payload_bytes())?;
            if tx.send(msg).is_err() {
                break;
            }
        }
        Ok::<_, Error>(())
    });
    Ok(rx)
}
