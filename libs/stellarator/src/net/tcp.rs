use crate::buf::{IoBuf, IoBufMut};
use crate::os::BorrowedHandle;
use crate::reactor::{ops, Completion};
use crate::Error;
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

    pub async fn read<B: IoBufMut>(&self, buf: B) -> Result<(usize, B), Error> {
        Completion::run(ops::Read::new(self.as_handle(), buf, None)).await
    }

    pub async fn write<B: IoBuf>(&self, buf: B) -> Result<(usize, B), Error> {
        Completion::run(ops::Write::new(self.as_handle(), buf, None)).await
    }

    fn as_handle(&self) -> BorrowedHandle<'_> {
        BorrowedHandle::Socket(&self.socket)
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
        let (_, socket) = Completion::run(op).await?;
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

    #[test]
    fn test_echo_server() {
        crate::test!(async {
            let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
            let addr = listener.local_addr().unwrap();
            let handle = crate::spawn(async move {
                let stream = listener.accept().await.unwrap();
                let buf = vec![0u8; 128];
                let (n, mut buf) = stream.read(buf).await.unwrap();
                buf.truncate(n);
                stream.write(buf).await.unwrap();
            });
            let stream = TcpStream::connect(addr).await.unwrap();
            stream.write(b"foo").await.unwrap();
            let buf = vec![0; 128];
            let (n, buf) = stream.read(buf).await.unwrap();
            handle.await.unwrap();
            assert_eq!(&buf[..n], b"foo");
        })
    }
}
