use fred::prelude::*;

const TOPIC_PING: &str = "ping";

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Ping {
    id: u32,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> RedisResult<()> {
    tracing_subscriber::fmt::init();

    let client = RedisClient::default();
    client.init().await?;
    let _: () = client.flushall(false).await?;

    // send ping
    let mq = redmq::MsgQueue::new(&client, "pinger", "").await?;
    mq.send(TOPIC_PING, vec![Ping { id: 74 }]).await?;

    // receive ping
    let msgs = mq.recv::<Ping>(TOPIC_PING, 1, None).await?;
    assert_eq!(msgs[0].id, 74);
    mq.ack(TOPIC_PING, &msgs).await?;
    mq.del(TOPIC_PING, &msgs).await?;

    Ok(())
}
