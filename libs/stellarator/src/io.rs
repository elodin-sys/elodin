use std::{collections::VecDeque, future::Future};

use crate::{
    buf::{self, IoBuf, IoBufMut, Slice},
    rent, BufResult, Error,
};

pub trait AsyncRead {
    fn read<B: IoBufMut>(&self, buf: B) -> impl Future<Output = BufResult<usize, B>>;
}

pub trait AsyncWrite {
    fn write<B: IoBuf>(&self, buf: B) -> impl Future<Output = BufResult<usize, B>>;

    fn write_all<B: IoBuf>(&self, mut buf: B) -> impl Future<Output = BufResult<(), B>> {
        async {
            let mut total_written = 0;
            while total_written < buf.init_len() {
                let slice = buf.try_slice(total_written..).expect("invalid slice");

                match self.write(slice).await {
                    (Ok(0), slice) => return (Err(Error::EOF), slice.into_inner()),
                    (Ok(n), slice) => {
                        total_written += n;
                        buf = slice.into_inner();
                    }
                    (Err(err), slice) => return (Err(err), slice.into_inner()),
                }
            }
            (Ok(()), buf)
        }
    }
}

pub struct LengthDelReader<A: AsyncRead> {
    reader: A,
    scratch: VecDeque<u8>,
}

impl<A: AsyncRead> LengthDelReader<A> {
    pub fn new(reader: A) -> Self {
        Self {
            reader,
            scratch: VecDeque::default(),
        }
    }

    async fn read<B: IoBufMut>(&mut self, mut buf: B) -> BufResult<usize, B> {
        if !self.scratch.is_empty() {
            let take_len = self.scratch.len().min(buf.total_len());
            let (front, back) = self.scratch.as_slices();
            let front_len = front.len().min(take_len);
            let back_len = (take_len - front_len).min(back.len());
            (buf::deref_mut(&mut buf)[..front_len]).copy_from_slice(&front[..front_len]);
            (buf::deref_mut(&mut buf)[front_len..back_len + front_len])
                .copy_from_slice(&back[..back_len]);
            self.scratch.drain(..take_len);
            (Ok(take_len), buf)
        } else {
            self.reader.read(buf).await
        }
    }

    pub async fn recv<B: IoBufMut>(&mut self, mut buf: B) -> Result<Slice<B>, Error> {
        let mut total_read = 0;
        while total_read < 8 {
            let mut slice = buf.try_slice(total_read..).ok_or(Error::BufferOverflow)?;
            total_read += rent!(self.read(slice).await, slice)?;
            buf = slice.into_inner();
        }
        let len_buf = &buf::deref(&buf)[..8];
        let len: [u8; 8] = len_buf[..8].try_into().expect("slice wasn't 8 long");
        let len = u64::from_le_bytes(len) as usize;
        let required_len = len.checked_add(8).ok_or(Error::IntegerOverflow)?;
        while total_read < required_len {
            let mut slice = buf.try_slice(total_read..).ok_or(Error::BufferOverflow)?;
            total_read += rent!(self.read(slice).await, slice)?;
            buf = slice.into_inner();
        }
        let extra_read = total_read.saturating_sub(required_len);
        if extra_read > 0 {
            let extra = &buf::deref(&buf)[required_len..total_read];
            self.scratch.extend(extra);
        }
        let buf = buf
            .try_slice(8..required_len)
            .ok_or(Error::BufferOverflow)?;
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use crate::net::{TcpListener, TcpStream};

    use super::*;

    #[test]
    fn test_length_del_tcp() {
        crate::test!(async {
            let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
            let addr = listener.local_addr().unwrap();
            let handle = crate::spawn(async move {
                let stream = listener.accept().await.unwrap();
                stream
                    .write(&[4, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4][..])
                    .await
                    .0
                    .unwrap();
                stream
                    .write(&[3, 0, 0, 0, 0, 0, 0, 0, 5][..])
                    .await
                    .0
                    .unwrap();
                stream.write(&[6, 7, 1, 0, 0][..]).await.0.unwrap();
                stream.write(&[0, 0, 0, 0, 0, 0xff][..]).await.0.unwrap();
            });
            let stream = TcpStream::connect(addr).await.unwrap();
            let mut stream = LengthDelReader::new(stream);

            let buf = vec![0u8; 64];
            let out = stream.recv(buf).await.unwrap();
            assert_eq!(&out[..], &[1, 2, 3, 4]);
            let buf = out.into_inner();
            let out = stream.recv(buf).await.unwrap();
            assert_eq!(&out[..], &[5, 6, 7]);
            let buf = out.into_inner();
            let out = stream.recv(buf).await.unwrap();
            assert_eq!(&out[..], &[0xff]);

            handle.await.unwrap();
        })
    }
}
