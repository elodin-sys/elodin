#![cfg(feature = "poll")]

use std::{net::SocketAddr, time::Duration};
use stellarator::net::{TcpListener, TcpStream, UdpSocket};

#[stellarator::test]
async fn pending_tcp_read_allows_timer_and_other_socket_work() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = stellarator::spawn(async move {
        let stream = listener.accept().await.unwrap();
        let (read, bytes) = stream.read(vec![0; 4]).await;
        assert_eq!(read.unwrap(), 4);
        assert_eq!(&bytes, b"ping");
        stellarator::sleep(Duration::from_millis(20)).await;
        stream.write(b"ready").await.0.unwrap();
    });
    let stream = TcpStream::connect(address).await.unwrap();
    stellarator::sleep(Duration::from_millis(20)).await;
    stream.write(b"ping").await.0.unwrap();
    let (read, bytes) = stream.read(vec![0; 5]).await;
    assert_eq!(read.unwrap(), 5);
    assert_eq!(&bytes, b"ready");
    server.await.unwrap();
}

#[stellarator::test]
async fn canceled_udp_read_releases_readiness_for_the_next_reader() {
    let receiver = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
    let address = receiver.local_addr().unwrap();
    futures_lite::future::or(
        async {
            receiver.recv(vec![0; 8]).await.0.unwrap();
            panic!("unexpected datagram");
        },
        stellarator::sleep(Duration::from_millis(20)),
    )
    .await;
    let sender = stellarator::spawn(async move {
        stellarator::sleep(Duration::from_millis(20)).await;
        let socket = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        socket.send_to(b"next", address).await.0.unwrap();
    });
    let (read, bytes) = receiver.recv(vec![0; 8]).await;
    assert_eq!(read.unwrap(), 4);
    assert_eq!(&bytes[..4], b"next");
    sender.await.unwrap();
}
