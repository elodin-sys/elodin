use roci::tcp::SinkExt;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use stellarator::io::SplitExt;
use stellarator::net::TcpStream;

async fn connect() -> anyhow::Result<()> {
    let stream = TcpStream::connect(SocketAddr::new([127, 0, 0, 1].into(), 2240))
        .await
        .map_err(anyhow::Error::from)?;
    let (rx, tx) = stream.split();
    let tx = impeller2_stella::PacketSink::new(tx);
    let _rx = impeller2_stella::PacketStream::new(rx);
    tx.init_msg::<Foo>().await?;
    tx.init_msg::<Bar>().await?;
    let mut flip = false;
    loop {
        tx.send(&Foo {
            bar: format!("{:?}", impeller2::types::Timestamp::now()),
            xyz: flip,
        })
        .await
        .0?;
        if flip {
            tx.send(&Bar {
                foo: Foo {
                    bar: format!("{:?}", std::time::Instant::now()),
                    xyz: flip,
                },
                test: 123,
            })
            .await
            .0?;
        }
        stellarator::sleep(std::time::Duration::from_millis(500)).await;
        flip = !flip;
    }
}

async fn run() -> anyhow::Result<()> {
    loop {
        if let Err(err) = connect().await {
            println!("error connecting {:?}", err);
            stellarator::sleep(std::time::Duration::from_millis(250)).await;
        }
    }
}

#[derive(Serialize, Deserialize, postcard_schema::Schema, Debug)]
struct Foo {
    bar: String,
    xyz: bool,
}

#[derive(Serialize, Deserialize, postcard_schema::Schema, Debug)]
struct Bar {
    foo: Foo,
    test: u32,
}

fn main() -> anyhow::Result<()> {
    stellarator::run(run)
}
