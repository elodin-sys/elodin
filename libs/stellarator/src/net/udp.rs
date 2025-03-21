use crate::BufResult;
use crate::buf::{IoBuf, IoBufMut};
use crate::io::{AsyncRead, AsyncWrite};
use crate::os::BorrowedHandle;
use crate::reactor::{Completion, ops};
use socket2::{SockAddr, Socket};
use std::io;
use std::net::SocketAddr;

pub struct UdpSocket {
    socket: Socket,
    connected_addr: Option<SocketAddr>,
}

impl UdpSocket {
    pub fn ephemeral() -> io::Result<UdpSocket> {
        let addr = SocketAddr::new([0; 4].into(), 0);
        Self::bind(addr)
    }

    pub fn bind(addr: SocketAddr) -> io::Result<UdpSocket> {
        let socket = socket2::Socket::new(
            socket2::Domain::for_address(addr),
            socket2::Type::DGRAM,
            None,
        )?;

        socket.set_reuse_address(true)?;
        #[cfg(unix)]
        socket.set_reuse_port(true)?;
        socket.set_nonblocking(!cfg!(target_os = "linux"))?;
        socket.bind(&addr.into())?;

        Ok(UdpSocket {
            socket,
            connected_addr: None,
        })
    }

    pub fn set_broadcast(&self, broadcast: bool) -> io::Result<()> {
        self.socket.set_broadcast(broadcast)
    }

    pub async fn recv<B: IoBufMut>(&self, buf: B) -> BufResult<usize, B> {
        Completion::run(ops::Read::new(self.as_handle(), buf, None)).await
    }

    pub async fn send<B: IoBuf>(&self, buf: B) -> BufResult<usize, B> {
        if let Some(addr) = &self.connected_addr {
            self.send_to(buf, *addr).await
        } else {
            Completion::run(ops::Write::new(self.as_handle(), buf, None)).await
        }
    }

    pub async fn send_to<B: IoBuf>(&self, buf: B, target: SocketAddr) -> BufResult<usize, B> {
        Completion::run(ops::SendTo::new(
            self.as_handle(),
            buf,
            Box::new(SockAddr::from(target).into()),
        ))
        .await
    }

    pub fn connect(&mut self, addr: SocketAddr) {
        self.connected_addr = Some(addr);
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket
            .local_addr()
            .map(|addr| addr.as_socket().unwrap())
    }

    fn as_handle(&self) -> BorrowedHandle<'_> {
        BorrowedHandle::Socket(&self.socket)
    }
}

impl AsyncRead for UdpSocket {
    fn read<B: IoBufMut>(&self, buf: B) -> impl std::future::Future<Output = BufResult<usize, B>> {
        self.recv(buf)
    }
}

impl AsyncWrite for UdpSocket {
    fn write<B: IoBuf>(&self, buf: B) -> impl std::future::Future<Output = BufResult<usize, B>> {
        self.send(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rent;

    #[test]
    fn test_p2p_send() {
        crate::test!(async {
            let mut a = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
            let mut b = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
            let a_addr = a.local_addr().unwrap();
            let b_addr = b.local_addr().unwrap();
            a.connect(b_addr);
            a.send(b"foo").await.0.unwrap();
            b.connect(a_addr);
            b.send(b"bar").await.0.unwrap();
            let mut out_buf = vec![0u8; 64];
            let n = rent!(a.recv(out_buf).await, out_buf).unwrap();
            assert_eq!(&out_buf[..n], b"bar");
            let mut out_buf = vec![0u8; 64];
            let n = rent!(b.recv(out_buf).await, out_buf).unwrap();
            assert_eq!(&out_buf[..n], b"foo");
        })
    }

    #[test]
    fn test_send_to_with_recv() {
        crate::test!(async {
            // Create two UDP sockets with dynamically assigned ports
            let a = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
            let b = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();

            // Get socket addresses
            let a_addr = a.local_addr().unwrap();
            let b_addr = b.local_addr().unwrap();

            // Send a message from socket A to socket B using send_to
            let (sent, _) = a.send_to(b"hello from a", b_addr).await;
            assert_eq!(sent.unwrap(), 12); // "hello from a" is 12 bytes

            // Receive the message on socket B using regular recv
            let mut recv_buf = vec![0u8; 64];
            let (received, buf) = b.recv(recv_buf).await;
            let received_len = received.unwrap();

            // Get the buffer back and verify its contents
            recv_buf = buf;
            assert_eq!(&recv_buf[..received_len], b"hello from a");

            // Now send from B to A using send_to
            let (sent, _) = b.send_to(b"response from b", a_addr).await;
            assert_eq!(sent.unwrap(), 15); // "response from b" is 15 bytes

            // Receive on A using regular recv
            let mut recv_buf = vec![0u8; 64];
            let (received, buf) = a.recv(recv_buf).await;
            let received_len = received.unwrap();

            // Verify the response
            recv_buf = buf;
            assert_eq!(&recv_buf[..received_len], b"response from b");
        })
    }
}
