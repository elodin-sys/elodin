const ENTRIES: usize = 1000;
const WORKERS: usize = 10;

const TOPIC_JOB: &str = "job";
const TOPIC_RESULTS: &str = "results";

#[derive(redmq::FromRedisValue, redmq::ToRedisArgs)]
pub struct Job {
    name: String,
    size: u32,
}

#[derive(redmq::FromRedisValue, redmq::ToRedisArgs, Clone)]
pub struct Results {
    name: String,
}

impl Job {
    fn new(i: u32) -> Self {
        Job {
            name: format!("job-{}", i),
            size: i,
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> redis::RedisResult<()> {
    tracing_subscriber::fmt::init();

    let redis_url = std::env::var("REDIS_URL").unwrap_or("redis://127.0.0.1/".to_string());
    let client = redis::Client::open(redis_url)?;

    for i in 0..WORKERS {
        let mut mq = redmq::MsgQueue::new(&client, "worker", i.to_string()).await?;
        let _: tokio::task::JoinHandle<redis::RedisResult<()>> = tokio::spawn(async move {
            tracing::info!(worker = i, "spawning");
            loop {
                let work = mq.recv::<Job>(TOPIC_JOB, 10).await?;
                let results = work
                    .iter()
                    .map(|w| w.name.clone())
                    .map(|name| Results { name })
                    .collect();
                mq.send(TOPIC_RESULTS, results).await?;
                mq.ack(TOPIC_JOB, &work).await?;
                mq.del(TOPIC_JOB, &work).await?;
            }
        });
    }

    let jobs = (0..ENTRIES as u32).map(Job::new).collect();

    let mut mq = redmq::MsgQueue::new(&client, "producer", "").await?;
    mq.send(TOPIC_JOB, jobs).await?;

    let mut num_results = 0;
    while num_results < ENTRIES {
        let results = mq.recv::<Results>(TOPIC_RESULTS, 100).await?;
        num_results += results.len();
        mq.ack(TOPIC_RESULTS, &results).await?;
        mq.del(TOPIC_RESULTS, &results).await?;
        tracing::debug!(total = num_results, "received results");
    }

    Ok(())
}
