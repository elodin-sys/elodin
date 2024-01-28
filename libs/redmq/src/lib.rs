// Redis-based message queue library

use std::collections::HashMap;

use redis::streams;
use redis::AsyncCommands;

pub use redis::RedisError;
pub use redis_derive::{FromRedisValue, ToRedisArgs};

mod adapter;

pub use adapter::StringAdapter;

pub struct MsgQueue {
    conn: redis::aio::MultiplexedConnection,
    // xread doesn't work well with multiplexed connections
    blocking_conn: redis::aio::Connection,
    group: String,
    consumer: String,
}

pub struct Received<M: Message> {
    id: String,
    entry: M,
}

type StreamRangeReply = Vec<HashMap<String, redis::Value>>;
type StreamReadReply = Vec<HashMap<String, Vec<HashMap<String, redis::Value>>>>;

impl MsgQueue {
    pub async fn new(
        client: &redis::Client,
        group: impl Into<String>,
        consumer: impl Into<String>,
    ) -> redis::RedisResult<Self> {
        Ok(Self {
            conn: client.get_multiplexed_async_connection().await?,
            blocking_conn: client.get_async_connection().await?,
            group: group.into(),
            consumer: consumer.into(),
        })
    }

    pub async fn send<M: Message>(
        &self,
        topic: &str,
        entries: Vec<M>,
    ) -> redis::RedisResult<Vec<String>> {
        entries
            .into_iter()
            .fold(&mut redis::pipe(), |p, e| p.xadd_map(topic, "*", e))
            .atomic()
            .query_async::<_, Vec<String>>(&mut self.conn.clone())
            .await
    }

    pub async fn get<M: Message>(&self, topic: &str, id: &str) -> redis::RedisResult<Option<M>> {
        self.conn
            .clone()
            .xrange::<_, _, _, StreamRangeReply>(topic, id, id)
            .await?
            .into_iter()
            .flat_map(|row| row.into_iter())
            .next()
            .map(|(_, v)| redis::from_redis_value(&v))
            .transpose()
    }

    // TODO(Akhil): Remove after Redis supports auto group creation (https://github.com/redis/redis/pull/10747)
    async fn register(&mut self, topic: &str) -> redis::RedisResult<()> {
        let result: Result<(), _> = self
            .blocking_conn
            .xgroup_create_mkstream(topic, &self.group, "0-0")
            .await;
        if let Err(err) = result {
            // ignore BUSYGROUP error, which indicates the group already exists
            if err.code().unwrap_or_default() != "BUSYGROUP" {
                return Err(err);
            }
        } else {
            tracing::debug!(group = &self.group, topic, "register")
        }
        Ok(())
    }

    // Will return immediately (without waiting for limit) if there is at least 1 message in the queue.
    pub async fn recv<M: Message>(
        &mut self,
        topic: &str,
        limit: usize,
    ) -> redis::RedisResult<Vec<Received<M>>> {
        // TODO(Akhil): add autoclaim (https://redis.io/docs/data-types/streams/#automatic-claiming)
        let mut check_pending = true;
        loop {
            let id = if check_pending { "0-0" } else { ">" };

            // TODO(Akhil): add jitter to block time to prevent thundering herd
            let opts = streams::StreamReadOptions::default()
                .block(1000)
                .count(limit)
                .group(&self.group, &self.consumer);
            let result = self
                .blocking_conn
                .xread_options::<_, _, StreamReadReply>(&[topic], &[id], &opts)
                .await;
            let srr = match result {
                Ok(srr) => srr,
                Err(err) => {
                    // create group if it doesn't exist, and retry the xread
                    if err.code().unwrap_or_default() == "NOGROUP" {
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
                group = &self.group,
                consumer = &self.consumer,
                start_id = id,
                count = stream.len(),
                "received"
            );

            if stream.is_empty() {
                // switch from pending (un ack'd) to new messages
                check_pending = false;
            }

            let mut bad_ids = Vec::default();
            let entries: Vec<Received<M>> = stream
                .into_iter()
                .filter_map(|(id, v)| match redis::from_redis_value(&v) {
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
                self.blocking_conn
                    .xack(topic, &self.group, &bad_ids)
                    .await?;
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
    ) -> redis::RedisResult<usize> {
        let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        let acks = self.conn.clone().xack(topic, &self.group, &ids).await?;
        tracing::trace!(
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
    ) -> redis::RedisResult<usize> {
        let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        let deleted = self.conn.clone().xdel(topic, &ids).await?;
        tracing::trace!(
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

pub trait Message: redis::FromRedisValue + redis::ToRedisArgs {}

impl<M: redis::FromRedisValue + redis::ToRedisArgs> Message for M {}
