use crate::buf::{IoBuf, IoBufMut};
use crate::io::{AsyncRead, AsyncWrite};
use crate::os::BorrowedHandle;
use crate::reactor::{ops, Completion};
use crate::{BufResult, Error};
use socket2::{SockAddr, Socket};
use std::io::{self};
use std::net::SocketAddr;

use super::SockAddrRaw;

pub struct TcpStream {
    socket: Socket,
}

impl TcpStream {
    pub async fn connect(addr: SocketAddr) -> Result<TcpStream, Error> {
        let socket = socket2::Socket::new(
            socket2::Domain::for_address(addr),
            socket2::Type::STREAM,
            Some(socket2::Protocol::TCP),
        )?;
        socket.set_nonblocking(true)?;
        let addr: SockAddr = addr.into();
        Completion::run(ops::Connect::new(&socket, Box::new(addr.into()))?).await?;

        Ok(TcpStream { socket })
    }

    pub async fn read<B: IoBufMut>(&self, buf: B) -> BufResult<usize, B> {
        Completion::run(ops::Read::new(self.as_handle(), buf, None)).await
    }

    pub async fn write<B: IoBuf>(&self, buf: B) -> BufResult<usize, B> {
        Completion::run(ops::Write::new(self.as_handle(), buf, None)).await
    }

    fn as_handle(&self) -> BorrowedHandle<'_> {
        BorrowedHandle::Socket(&self.socket)
    }
}

impl AsyncRead for TcpStream {
    fn read<B: IoBufMut>(&self, buf: B) -> impl std::future::Future<Output = BufResult<usize, B>> {
        self.read(buf)
    }
}

impl AsyncWrite for TcpStream {
    fn write<B: IoBuf>(&self, buf: B) -> impl std::future::Future<Output = BufResult<usize, B>> {
        self.write(buf)
    }
}

pub struct TcpListener {
    socket: socket2::Socket,
}

impl TcpListener {
    pub fn bind(addr: SocketAddr) -> Result<TcpListener, Error> {
        let socket = socket2::Socket::new(
            socket2::Domain::for_address(addr),
            socket2::Type::STREAM,
            Some(socket2::Protocol::TCP),
        )?;

        socket.bind(&addr.into())?;
        socket.listen(1024)?;
        socket.set_nonblocking(true)?;

        Ok(TcpListener { socket })
    }

    pub async fn accept(&self) -> Result<TcpStream, Error> {
        let op = ops::Accept::new(&self.socket, Box::new(SockAddrRaw::zeroed()));
        let socket = Completion::run(op).await.0?;
        Ok(TcpStream { socket })
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket
            .local_addr()
            .map(|addr| addr.as_socket().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rent;

    #[test]
    fn test_echo_server() {
        crate::test!(async {
            let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
            let addr = listener.local_addr().unwrap();
            let handle = crate::spawn(async move {
                let stream = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 128];
                let n = rent!(stream.read(buf).await, buf).unwrap();
                buf.truncate(n);
                stream.write(buf).await.0.unwrap();
            });
            let stream = TcpStream::connect(addr).await.unwrap();
            stream.write(b"foo").await.0.unwrap();
            let mut buf = vec![0; 128];
            let n = rent!(stream.read(buf).await, buf).unwrap();
            assert_eq!(&buf[..n], b"foo");
            handle.await.unwrap();
        })
    }
}
