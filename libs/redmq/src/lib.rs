// Redis-based message queue library

use std::time::Duration;
use std::{collections::HashMap, fmt};

use fred::prelude::*;

mod de;
mod ser;

pub use de::from_redis;
pub use ser::to_redis;

pub struct MsgQueue {
    client: RedisClient,
    group: String,
    consumer: String,
}

pub struct Received<M: Message> {
    id: String,
    entry: M,
}

#[derive(Debug)]
enum Error {
    Message(String),
    UnsupportedType,
    FailedToParse,
}

impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Message(msg) => formatter.write_str(msg),
            Error::UnsupportedType => formatter.write_str("unsupported type"),
            Error::FailedToParse => formatter.write_str("failed to parse"),
        }
    }
}

impl serde::ser::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Error::Message(msg.to_string())
    }
}

impl serde::de::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Error::Message(msg.to_string())
    }
}

type StreamReadReply = Vec<HashMap<String, Vec<HashMap<String, Vec<(String, String)>>>>>;

impl MsgQueue {
    pub async fn new(
        client: &RedisClient,
        group: impl Into<String>,
        consumer: impl Into<String>,
    ) -> RedisResult<Self> {
        let client = client.clone_new();
        client.init().await?;
        Ok(Self {
            client,
            group: group.into(),
            consumer: consumer.into(),
        })
    }

    pub async fn send<M: Message>(&self, topic: &str, entries: Vec<M>) -> RedisResult<Vec<String>> {
        let pipeline = self.client.pipeline();
        for entry in entries {
            let _: () = pipeline
                .xadd(topic, false, None, "*", to_redis(&entry))
                .await?;
        }
        let ids: Vec<String> = pipeline.all().await?;
        tracing::trace!(
            topic,
            group = &self.group,
            consumer = &self.consumer,
            count = ids.len(),
            "sent"
        );
        Ok(ids)
    }

    async fn register(&self, topic: &str) -> RedisResult<()> {
        let result: Result<(), _> = self
            .client
            .xgroup_create(topic, &self.group, "0-0", true)
            .await;
        if let Err(err) = result {
            // ignore BUSYGROUP error, which indicates the group already exists
            tracing::warn!(consumer = &self.consumer, ?err);
            if !err.details().starts_with("BUSYGROUP") {
                return Err(err);
            }
        } else {
            tracing::debug!(group = &self.group, topic, "register")
        }
        Ok(())
    }

    pub async fn recv<M: Message>(
        &self,
        topic: &str,
        count: usize,
        last_id: Option<&str>,
    ) -> RedisResult<Vec<Received<M>>> {
        // TODO(Akhil): add autoclaim (https://redis.io/docs/data-types/streams/#automatic-claiming)
        let mut check_pending = true;
        let mut delay = Duration::from_secs(1);
        loop {
            let id = if check_pending {
                last_id.unwrap_or("0-0")
            } else {
                ">"
            };
            // exponential backoff with a max of 2 minutes
            delay = Duration::from_secs(2 * 60).min(delay * 2);

            let result: Result<StreamReadReply, _> = self
                .client
                .xreadgroup(
                    &self.group,
                    &self.consumer,
                    Some(count as u64),
                    Some(delay.as_millis() as u64),
                    false,
                    &[topic],
                    id,
                )
                .await;
            let srr = match result {
                Ok(srr) => srr,
                Err(err) => {
                    // create group if it doesn't exist, and retry the xread
                    tracing::warn!(consumer = &self.consumer, ?err);
                    if err.details().starts_with("NOGROUP") {
                        self.register(topic).await?;
                        continue;
                    }
                    return Err(err);
                }
            };
            let stream: Vec<_> = srr
                .into_iter()
                .next()
                .unwrap_or_default()
                .remove(topic)
                .unwrap_or_default()
                .into_iter()
                .flat_map(|row| row.into_iter())
                .collect();

            tracing::trace!(
                topic,
                group = &self.group,
                consumer = &self.consumer,
                start_id = id,
                count = stream.len(),
                ?delay,
                "received"
            );

            if stream.is_empty() {
                // switch from pending (un ack'd) to new messages
                check_pending = false;
            }

            let mut bad_ids = Vec::default();
            let entries: Vec<Received<M>> = stream
                    .into_iter()
                    .filter_map(|(id, v)| match from_redis(v) {
                        Ok(entry) => Some(Received { id, entry }),
                        Err(err) => {
                            tracing::error!(consumer = &self.consumer, id, %err, "failed to parse message");
                            bad_ids.push(id);
                            None
                        }
                    })
                    .collect();

            if !bad_ids.is_empty() {
                // auto-ack bad messages
                let _: () = self.client.xack(topic, &self.group, bad_ids).await?;
            }

            if !entries.is_empty() {
                return Ok(entries);
            }
        }
    }

    pub async fn ack<M: Message>(
        &self,
        topic: &str,
        entries: &[Received<M>],
    ) -> RedisResult<usize> {
        let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        let acks = self.client.xack(topic, &self.group, ids).await?;
        tracing::trace!(
            topic,
            group = &self.group,
            consumer = &self.consumer,
            count = acks,
            "ack'd"
        );
        Ok(acks)
    }

    pub async fn del<M: Message>(
        &self,
        topic: &str,
        entries: &[Received<M>],
    ) -> RedisResult<usize> {
        let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        let deleted = self.client.xdel(topic, ids).await?;
        tracing::trace!(
            topic,
            group = &self.group,
            consumer = &self.consumer,
            count = deleted,
            "deleted"
        );
        Ok(deleted)
    }
}

impl<M: Message> std::ops::Deref for Received<M> {
    type Target = M;
    fn deref(&self) -> &Self::Target {
        &self.entry
    }
}

impl<M: Message> Received<M> {
    pub fn id(&self) -> &str {
        &self.id
    }
}

pub trait Message: serde::Serialize + serde::de::DeserializeOwned {}

impl<M: serde::Serialize + serde::de::DeserializeOwned> Message for M {}

#[cfg(test)]
mod tests {
    use crate::{from_redis, to_redis};

    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Debug)]
    struct Data {
        name: String,
        samples: usize,
    }

    #[test]
    fn simple_data() {
        let data = Data {
            name: "test".to_string(),
            samples: 774,
        };
        let redis_data = to_redis(&data);
        assert_eq!(redis_data.len(), 2);

        let deserialized_data: Data = from_redis(redis_data).unwrap();
        assert_eq!(data, deserialized_data);
    }
}
