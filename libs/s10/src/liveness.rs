use std::{
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    time::Duration,
};

use stellarator::{io::AsyncRead, net};
use tokio::io::AsyncWriteExt;

pub async fn serve_tokio() -> std::io::Result<u16> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;
    let port = local_addr.port();
    tokio::spawn(async move {
        tracing::info!("Serving liveness on {}", local_addr);
        loop {
            let mut stream = match listener.accept().await {
                Ok((stream, _)) => stream,
                Err(err) => {
                    tracing::error!("Error accepting connection: {}", err);
                    break;
                }
            };
            tokio::spawn(async move {
                let mut i = 0u64;
                loop {
                    let buf = i.to_be_bytes().to_vec();
                    let result = stream.write_all(&buf).await;
                    if let Err(err) = result {
                        tracing::error!("Error writing to socket: {}", err);
                        break;
                    }
                    i += 1;
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            });
        }
    });
    Ok(port)
}

pub fn monitor(port: u16) {
    stellarator::spawn(async move {
        let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
        tracing::info!("Monitoring liveness on {}", addr);
        let stream = match net::TcpStream::connect(addr).await {
            Ok(stream) => stream,
            Err(err) => {
                tracing::error!("Error connecting to liveness server: {}", err);
                std::process::exit(1);
            }
        };
        let mut buf = vec![0u8; 8];
        loop {
            if stellarator::rent!(stream.read_exact(buf).await, buf).is_err() {
                tracing::info!("Liveness server disconnected, terminating");
                std::process::exit(0);
            }
            let i = u64::from_be_bytes(buf.as_slice().try_into().unwrap());
            tracing::trace!("Received {}", i);
        }
    });
}
