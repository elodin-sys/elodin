const TOPIC_PING: &str = "ping";

#[derive(redmq::FromRedisValue, redmq::ToRedisArgs)]
pub struct Ping {
    id: u32,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> redis::RedisResult<()> {
    tracing_subscriber::fmt::init();

    let redis_url = std::env::var("REDIS_URL").unwrap_or("redis://127.0.0.1/".to_string());
    let client = redis::Client::open(redis_url)?;

    // send ping
    let mut mq = redmq::MsgQueue::new(&client, "pinger", "").await?;
    mq.send(TOPIC_PING, vec![Ping { id: 74 }]).await?;

    // receive ping
    let msgs = mq.recv::<Ping>(TOPIC_PING, 1).await?;
    assert_eq!(msgs[0].id, 74);
    mq.ack(TOPIC_PING, &msgs).await?;
    mq.del(TOPIC_PING, &msgs).await?;

    Ok(())
}
