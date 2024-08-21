use bytes::{Buf, BufMut};

use crate::{ser_de::Slice, Error, Packet, Payload};

pub struct Client<T> {
    inner: T,
}

impl<T> Client<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T> Client<T>
where
    T: embedded_io_async::Write,
{
    pub async fn send(&mut self, packet: Packet<Payload<impl Buf + Slice>>) -> Result<(), Error> {
        let mut buf = [0u8; 1024];
        let mut writer = Cursor::new(&mut buf[4..]);
        packet.write(&mut writer)?;
        let len = writer.pos;
        let len_bytes = (writer.pos as u32).to_be_bytes();
        buf[..4].copy_from_slice(&len_bytes);
        self.inner
            .write_all(&buf[..len + 4])
            .await
            .map_err(|_| Error::EOF)?;
        Ok(())
    }
}

struct Cursor<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
}

unsafe impl BufMut for Cursor<'_> {
    fn remaining_mut(&self) -> usize {
        self.buf.len() - self.pos
    }

    unsafe fn advance_mut(&mut self, cnt: usize) {
        self.pos += cnt;
    }

    fn chunk_mut(&mut self) -> &mut bytes::buf::UninitSlice {
        bytes::buf::UninitSlice::new(&mut self.buf[self.pos..])
    }
}
