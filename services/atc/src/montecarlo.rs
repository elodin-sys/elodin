pub const BATCH_SIZE: usize = 100;
pub const MAX_SAMPLE_COUNT: usize = 100_000;

pub const RUN_TOPIC: &str = "mc:run";

// Buffer batches help with work alignment
// E.g. if an agent can do 10 batches at once, adding some buffer between runs allows for the agent
// to only work on 1 run at a time. These batches are effectively a no-op, and are ignored during
// results aggregation.
// pub const BUFFER_BATCH_COUNT: usize = 10;

#[derive(redmq::FromRedisValue, redmq::ToRedisArgs)]
pub struct Run {
    // TODO(Akhil): remove string adapter after there's a release with https://github.com/redis-rs/redis-rs/pull/1029
    pub id: redmq::StringAdapter<uuid::Uuid>,
    pub name: String,
    pub samples: usize,
    pub batch_size: usize,
    pub start_time: redmq::StringAdapter<chrono::DateTime<chrono::Utc>>,
}

#[derive(redmq::FromRedisValue, redmq::ToRedisArgs)]
pub struct Batch {
    pub id: String,
    pub batch_no: usize,
    pub buffer: bool,
}
