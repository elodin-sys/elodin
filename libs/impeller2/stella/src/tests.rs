use super::*;
use impeller2::types::MsgExt;
use impeller2::types::{Msg, PacketId};
use postcard::experimental::max_size::MaxSize;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use stellarator::net::{TcpListener, TcpStream};

#[derive(Serialize, Deserialize, MaxSize, PartialEq, Debug)]
struct Foo {
    bar: u32,
}

impl Msg for Foo {
    const ID: PacketId = [0x1, 0x2];
}

#[test]
fn test_packet_echo() {
    stellarator::test!(async {
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        let addr = listener.local_addr().unwrap();
        stellarator::spawn(async move {
            let sink = PacketSink::new(listener.accept().await.unwrap());
            let msg = Foo { bar: 0xBB }.to_len_packet();
            sink.send(msg).await.0.unwrap();
        });
        let stream = TcpStream::connect(addr).await.unwrap();
        let mut stream = PacketStream::new(stream);
        let buf = vec![0; 128];
        let OwnedPacket::Msg(m) = stream.next(buf).await.unwrap() else {
            panic!("non msg pkt");
        };
        assert_eq!(m.id, Foo::ID);
        let foo: Foo = m.parse().unwrap();
        assert_eq!(foo, Foo { bar: 0xBB });
    })
}
