use std::{collections::VecDeque, future::Future, marker::PhantomData};

use crate::{
    BufResult, Error,
    buf::{self, IoBuf, IoBufMut, Slice},
    rent,
};
use std::sync::Arc;

pub trait AsyncRead {
    fn read<B: IoBufMut>(&self, buf: B) -> impl Future<Output = BufResult<usize, B>>;

    fn read_exact<B: IoBufMut>(&self, mut buf: B) -> impl Future<Output = BufResult<(), B>> {
        async {
            let mut total_read = 0;
            while total_read < buf.init_len() {
                let slice = buf.try_slice(total_read..).expect("invalid slice");

                match self.read(slice).await {
                    (Ok(0), slice) => return (Err(Error::EOF), slice.into_inner()),
                    (Ok(n), slice) => {
                        total_read += n;
                        buf = slice.into_inner();
                    }
                    (Err(err), slice) => return (Err(err), slice.into_inner()),
                }
            }
            (Ok(()), buf)
        }
    }
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

pub struct LengthDelReader<A: AsyncRead, L = u32> {
    reader: A,
    scratch: VecDeque<u8>,
    phantom_data: PhantomData<L>,
}

impl<A: AsyncRead, L: Length> LengthDelReader<A, L> {
    pub fn new(reader: A) -> Self {
        Self {
            reader,
            scratch: VecDeque::default(),
            phantom_data: PhantomData,
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

    pub async fn recv<B: IoBufMut>(&mut self, buf: B) -> Result<Slice<B>, Error> {
        let slice = self.recv_growable(GrowableBufWrapper(buf)).await?;
        let range = slice.range();
        let GrowableBufWrapper(inner) = slice.into_inner();
        Ok(unsafe { Slice::new_unchecked(inner, range) })
    }

    pub async fn recv_growable<B: IoBufMut + GrowableBuf>(
        &mut self,
        mut buf: B,
    ) -> Result<Slice<B>, Error> {
        let mut total_read = 0;
        while total_read < size_of::<L>() {
            let mut slice = buf
                .try_slice(total_read..)
                .ok_or(Error::BufferOverflow)
                .unwrap();
            let len = rent!(self.read(slice).await, slice)?;
            if len == 0 {
                return Err(Error::EOF);
            }
            total_read += len;
            buf = slice.into_inner();
        }
        let len_buf = &buf::deref(&buf)[..L::SIZE];
        let len: L::Buf = len_buf[..L::SIZE]
            .try_into()
            .map_err(|_| ())
            .expect("slice wasn't L::SIZE long");
        let len = L::from_le_bytes(len);
        let len = len.as_usize();
        let required_len = len.checked_add(L::SIZE).ok_or(Error::IntegerOverflow)?;
        buf.grow(required_len);
        while total_read < required_len {
            let mut slice = buf.try_slice(total_read..).ok_or(Error::BufferOverflow)?;
            let read_len = rent!(self.read(slice).await, slice)?;
            if read_len == 0 {
                return Err(Error::EOF);
            }
            total_read += read_len;
            buf = slice.into_inner();
        }
        let extra_read = total_read.saturating_sub(required_len);
        if extra_read > 0 {
            let extra = &buf::deref(&buf)[required_len..total_read];
            self.scratch.extend(extra);
        }
        let buf = buf
            .try_slice(L::SIZE..required_len)
            .ok_or(Error::BufferOverflow)?;
        Ok(buf)
    }
}

pub trait GrowableBuf {
    fn grow(&mut self, new_len: usize);
}

impl GrowableBuf for Vec<u8> {
    fn grow(&mut self, new_len: usize) {
        if new_len > self.len() {
            self.resize(new_len, 0);
        }
    }
}

#[derive(Clone)]
struct GrowableBufWrapper<B>(B);

unsafe impl<B: IoBuf> IoBuf for GrowableBufWrapper<B> {
    fn stable_init_ptr(&self) -> *const u8 {
        self.0.stable_init_ptr()
    }

    fn init_len(&self) -> usize {
        self.0.init_len()
    }

    fn total_len(&self) -> usize {
        self.0.total_len()
    }
}

unsafe impl<B: IoBufMut> IoBufMut for GrowableBufWrapper<B> {
    fn stable_mut_ptr(&mut self) -> std::ptr::NonNull<std::mem::MaybeUninit<u8>> {
        self.0.stable_mut_ptr()
    }

    unsafe fn set_init(&mut self, len: usize) {
        // safety: same safety guarantees as the wrapped buffer
        unsafe {
            self.0.set_init(len);
        }
    }
}

impl<B> GrowableBuf for GrowableBufWrapper<B> {
    fn grow(&mut self, _new_len: usize) {}
}

pub trait Length {
    const SIZE: usize;
    type Buf: for<'a> TryFrom<&'a [u8]>;
    fn as_usize(&self) -> usize;
    fn from_le_bytes(buf: Self::Buf) -> Self;
}

impl Length for u64 {
    const SIZE: usize = 8;
    type Buf = [u8; 8];
    fn as_usize(&self) -> usize {
        *self as usize
    }

    fn from_le_bytes(buf: Self::Buf) -> Self {
        Self::from_le_bytes(buf)
    }
}

impl Length for u32 {
    const SIZE: usize = 4;
    type Buf = [u8; 4];
    fn as_usize(&self) -> usize {
        *self as usize
    }

    fn from_le_bytes(buf: Self::Buf) -> Self {
        Self::from_le_bytes(buf)
    }
}

pub struct OwnedReader<R> {
    inner: Arc<R>,
}

impl<R: AsyncRead> AsyncRead for OwnedReader<R> {
    fn read<B: IoBufMut>(&self, buf: B) -> impl Future<Output = BufResult<usize, B>> {
        self.inner.read(buf)
    }
}

pub struct OwnedWriter<W> {
    inner: Arc<W>,
}

impl<W: AsyncWrite> AsyncWrite for OwnedWriter<W> {
    fn write<B: IoBuf>(&self, buf: B) -> impl Future<Output = BufResult<usize, B>> {
        self.inner.write(buf)
    }
}

pub trait SplitExt: AsyncRead + AsyncWrite + Sized {
    fn split(self) -> (OwnedReader<Self>, OwnedWriter<Self>)
    where
        Self: Sized,
    {
        let arc = Arc::new(self);
        (
            OwnedReader { inner: arc.clone() },
            OwnedWriter { inner: arc },
        )
    }
}

impl<T: AsyncRead + AsyncWrite> SplitExt for T {}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use crate::net::{TcpListener, TcpStream};
    use crate::test;

    use super::*;

    #[test]
    async fn test_length_del_tcp() {
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = crate::spawn(async move {
            let stream = listener.accept().await.unwrap();
            stream.write(&[4, 0, 0, 0, 1, 2, 3, 4][..]).await.0.unwrap();
            stream.write(&[3, 0, 0, 0, 5][..]).await.0.unwrap();
            stream.write(&[6, 7, 1, 0][..]).await.0.unwrap();
            stream.write(&[0, 0, 0xff][..]).await.0.unwrap();
        });
        let stream = TcpStream::connect(addr).await.unwrap();
        let mut stream = LengthDelReader::<_, u32>::new(stream);

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
    }

    #[test]
    async fn test_length_del_tcp_u64() {
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
        let mut stream = LengthDelReader::<_, u64>::new(stream);

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
    }
}
