use crate::buf::{IoBuf, IoBufMut};
use crate::os::BorrowedHandle;
use crate::reactor::{ops, Completion};
use crate::Error;
use socket2::{SockAddr, Socket};
use std::io;
use std::net::SocketAddr;

pub struct UdpSocket {
    socket: Socket,
}

impl UdpSocket {
    pub fn bind(addr: SocketAddr) -> io::Result<UdpSocket> {
        let socket = socket2::Socket::new(
            socket2::Domain::for_address(addr),
            socket2::Type::DGRAM,
            None,
        )?;

        socket.set_reuse_address(true)?;
        socket.set_nonblocking(true)?;
        socket.bind(&addr.into())?;

        Ok(UdpSocket { socket })
    }

    pub async fn recv<B: IoBufMut>(&self, buf: B) -> Result<(usize, B), Error> {
        Completion::run(ops::Read::new(self.as_handle(), buf, None)).await
    }

    pub async fn send<B: IoBuf>(&self, buf: B) -> Result<(usize, B), Error> {
        Completion::run(ops::Write::new(self.as_handle(), buf, None)).await
    }

    pub async fn connect(&self, addr: SocketAddr) -> Result<(), Error> {
        let addr: SockAddr = addr.into();
        Completion::run(ops::Connect::new(&self.socket, Box::new(addr.into()))?).await?;
        Ok(())
    }

    // TODO: add these back when we add UD `recv_from` and `send_to` support to reactors
    // pub async fn recv_from<B: IoBufMut>(&self, buf: B) -> Result<(usize, SocketAddr, B), Error> {
    //     let op = crate::reactor::ops::RecvFrom::new(
    //         self.as_handle(),
    //         buf,
    //         Box::new(SockAddrRaw::zeroed()),
    //     );
    //     let (n, addr, buf) = Completion::run(op).await?;
    //     let addr = socket2::SockAddr::from(*addr);
    //     let addr = addr.as_socket().ok_or(Error::InvalidSocketAddrType)?;
    //     Ok((n, addr, buf))
    // }

    // pub async fn send_to<B: IoBuf>(&self, buf: B, target: SocketAddr) -> Result<(usize, B), Error> {
    //     let addr = SockAddrRaw::from(target);
    //     Completion::run(crate::reactor::ops::SendTo::new(
    //         self.as_handle(),
    //         buf,
    //         Box::new(addr),
    //     ))
    //     .await
    // }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket
            .local_addr()
            .map(|addr| addr.as_socket().unwrap())
    }

    fn as_handle(&self) -> BorrowedHandle<'_> {
        BorrowedHandle::Socket(&self.socket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p2p_send() {
        crate::test!(async {
            let a = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
            let b = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
            let a_addr = a.local_addr().unwrap();
            let b_addr = b.local_addr().unwrap();
            a.connect(b_addr).await.unwrap();
            a.send(b"foo").await.unwrap();
            b.connect(a_addr).await.unwrap();
            b.send(b"bar").await.unwrap();
            let out_buf = vec![0u8; 64];
            let (n, out_buf) = a.recv(out_buf).await.unwrap();
            assert_eq!(&out_buf[..n], b"bar");
            let out_buf = vec![0u8; 64];
            let (n, out_buf) = b.recv(out_buf).await.unwrap();
            assert_eq!(&out_buf[..n], b"foo");
        })
    }
}
