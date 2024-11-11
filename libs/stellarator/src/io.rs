use std::future::Future;

use crate::{
    buf::{IoBuf, IoBufMut},
    BufResult,
};

pub trait AsyncRead {
    fn read<B: IoBufMut>(&self, buf: B) -> impl Future<Output = BufResult<usize, B>>;
}

pub trait AsyncWrite {
    fn write<B: IoBuf>(&self, buf: B) -> impl Future<Output = BufResult<usize, B>>;
}
