pub const BATCH_SIZE: usize = 100;
pub const MAX_SAMPLE_COUNT: usize = 100_000;

pub const RUN_TOPIC: &str = "mc:run";

// Buffer batches help with work alignment
// E.g. if an agent can do 10 batches at once, adding some buffer between runs allows for the agent
// to only work on 1 run at a time. These batches are effectively a no-op, and are ignored during
// results aggregation.
// pub const BUFFER_BATCH_COUNT: usize = 10;

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Debug)]
pub struct Run {
    pub id: uuid::Uuid,
    pub name: String,
    pub samples: usize,
    pub batch_size: usize,
    pub start_time: chrono::DateTime<chrono::Utc>,
}

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Debug)]
pub struct Batch {
    pub id: String,
    pub batch_no: usize,
    pub buffer: bool,
}

#[cfg(test)]
mod tests {
    use super::{Batch, Run};

    #[test]
    fn ser_de_run() {
        let run = Run {
            id: uuid::Uuid::now_v7(),
            name: "test_run".to_string(),
            samples: 774,
            batch_size: 100,
            start_time: chrono::Utc::now(),
        };

        let run_de = redmq::from_redis::<Run>(redmq::to_redis(&run)).unwrap();
        assert_eq!(run_de, run);
    }

    #[test]
    fn ser_de_batch() {
        let batch = Batch {
            id: "0-0".to_string(),
            batch_no: 6,
            buffer: false,
        };

        let batch_de = redmq::from_redis::<Batch>(redmq::to_redis(&batch)).unwrap();
        assert_eq!(batch_de, batch);
    }
}
