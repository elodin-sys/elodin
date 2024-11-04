use super::OpCode;
use crate::buf::IoBuf;
use crate::buf::IoBufMut;
use crate::net::SockAddrRaw;
use crate::os::{BorrowedHandle, OwnedHandle};
use crate::Error;
use io_uring::types::{TimeoutFlags, Timespec};
use io_uring::{cqueue, opcode, squeue, types};
use socket2::Socket;
use std::time::Duration;
use std::{
    ffi::CString,
    os::{
        fd::{AsFd, AsRawFd, FromRawFd, OwnedFd, RawFd},
        unix::ffi::OsStrExt,
    },
    path::PathBuf,
};

pub struct Nop;

impl OpCode for Nop {
    type Output = ();

    fn output(self, entry: cqueue::Entry) -> Result<Self::Output, Error> {
        entry.as_result().map(|_| ())
    }

    unsafe fn sqe(&mut self) -> squeue::Entry {
        io_uring::opcode::Nop::new().build()
    }

    type Buf = ();

    fn into_buf(self) -> Self::Buf {}
}

pub(crate) trait CqeExt {
    fn as_result(&self) -> Result<u32, Error>;
}

impl CqeExt for cqueue::Entry {
    fn as_result(&self) -> Result<u32, Error> {
        let res = self.result();
        if res >= 0 {
            Ok(res as u32)
        } else {
            Err(Error::Io(std::io::Error::from_raw_os_error(-res)))
        }
    }
}

pub struct Read<'fd, T> {
    fd: BorrowedHandle<'fd>,
    buf: T,
    offset: Option<u64>,
}

impl<'fd, T: IoBufMut> OpCode for Read<'fd, T> {
    type Output = (usize, T);

    fn output(self, entry: cqueue::Entry) -> Result<Self::Output, crate::Error> {
        entry.as_result().map(|res| (res as usize, self.buf))
    }

    unsafe fn sqe(&mut self) -> squeue::Entry {
        let ptr = self.buf.stable_mut_ptr().as_ptr() as *mut u8;
        let len = self.buf.total_len() as u32;

        let mut sqe = opcode::Read::new(types::Fd(self.fd.as_raw_fd()), ptr, len);
        if let Some(offset) = self.offset {
            sqe = sqe.offset(offset);
        }
        sqe.build()
    }

    type Buf = T;

    fn into_buf(self) -> Self::Buf {
        self.buf
    }
}

impl<'fd, T> Read<'fd, T> {
    pub fn new(fd: BorrowedHandle<'fd>, buf: T, offset: Option<u64>) -> Self {
        Self { fd, buf, offset }
    }
}

pub struct Write<'fd, T> {
    fd: BorrowedHandle<'fd>,
    buf: T,
    offset: Option<u64>,
}

impl<'fd, T: IoBuf> OpCode for Write<'fd, T> {
    type Output = (usize, T);

    fn output(self, entry: cqueue::Entry) -> Result<Self::Output, crate::Error> {
        entry.as_result().map(|res| (res as usize, self.buf))
    }

    unsafe fn sqe(&mut self) -> squeue::Entry {
        let ptr = self.buf.stable_init_ptr();
        let len = self.buf.init_len() as u32;

        let mut sqe = opcode::Write::new(types::Fd(self.fd.as_raw_fd()), ptr, len);
        if let Some(offset) = self.offset {
            sqe = sqe.offset(offset);
        }
        sqe.build()
    }

    type Buf = T;

    fn into_buf(self) -> Self::Buf {
        self.buf
    }
}

impl<'fd, T> Write<'fd, T> {
    pub fn new(fd: BorrowedHandle<'fd>, buf: T, offset: Option<u64>) -> Self {
        Self { fd, buf, offset }
    }
}

pub struct Open {
    path: CString,
    flags: i32,
    mode: libc::mode_t,
}

impl OpCode for Open {
    type Output = OwnedHandle;

    fn output(self, entry: cqueue::Entry) -> Result<Self::Output, crate::Error> {
        let fd = entry.as_result()? as RawFd;
        // safety: io_uring has just given a file-descriptor and we are the sole owner
        let fd = unsafe { OwnedHandle::Fd(OwnedFd::from_raw_fd(fd)) };
        Ok(fd)
    }

    unsafe fn sqe(&mut self) -> squeue::Entry {
        opcode::OpenAt::new(types::Fd(libc::AT_FDCWD), self.path.as_ptr())
            .flags(self.flags)
            .mode(self.mode)
            .build()
    }

    type Buf = ();

    fn into_buf(self) -> Self::Buf {}
}

impl Open {
    pub fn new(path: PathBuf, options: &crate::fs::OpenOptions) -> Result<Self, Error> {
        let flags = libc::O_CLOEXEC
            | options.access_mode()?
            | options.creation_mode()?
            | options.custom_flags;
        Ok(Self {
            path: CString::new(path.as_os_str().as_bytes()).map_err(std::io::Error::from)?,
            flags,
            mode: options.mode,
        })
    }
}

pub struct Close {
    raw_fd: RawFd,
}

impl OpCode for Close {
    type Output = ();

    fn output(self, entry: cqueue::Entry) -> Result<Self::Output, crate::Error> {
        entry.as_result().map(|_| ())
    }

    unsafe fn sqe(&mut self) -> squeue::Entry {
        opcode::Close::new(types::Fd(self.raw_fd)).build()
    }

    type Buf = ();

    fn into_buf(self) -> Self::Buf {}
}

impl Close {
    pub fn new(fd: OwnedHandle) -> Self {
        let fd = fd.as_fd().as_raw_fd();
        Self { raw_fd: fd }
    }
}

pub struct Accept<'fd> {
    fd: &'fd Socket,
    sock_addr: Box<SockAddrRaw>,
    flags: i32,
}

impl<'fd> OpCode for Accept<'fd> {
    type Output = (Box<SockAddrRaw>, Socket);

    fn output(self, entry: cqueue::Entry) -> Result<Self::Output, Error> {
        let fd = entry.as_result()? as RawFd;
        // safety: io_uring has just given a file-descriptor and we are the sole owner
        let fd = unsafe { Socket::from_raw_fd(fd) };
        Ok((self.sock_addr, fd))
    }

    unsafe fn sqe(&mut self) -> squeue::Entry {
        opcode::Accept::new(
            types::Fd(self.fd.as_raw_fd()),
            &raw mut self.sock_addr.storage as _,
            &raw mut self.sock_addr.len,
        )
        .flags(self.flags)
        .build()
    }

    type Buf = Box<SockAddrRaw>;

    fn into_buf(self) -> Self::Buf {
        self.sock_addr
    }
}

impl<'fd> Accept<'fd> {
    pub fn new(fd: &'fd Socket, sock_addr: Box<SockAddrRaw>) -> Self {
        Self {
            fd,
            sock_addr,
            flags: 0,
        }
    }

    pub fn flags(mut self, flags: i32) -> Self {
        self.flags = flags;
        self
    }
}

pub struct Connect<'fd> {
    fd: &'fd Socket,
    sock_addr: Box<SockAddrRaw>,
}

impl<'fd> OpCode for Connect<'fd> {
    type Output = ();

    fn output(self, entry: cqueue::Entry) -> Result<Self::Output, Error> {
        entry.as_result().map(|_| ())
    }

    unsafe fn sqe(&mut self) -> squeue::Entry {
        opcode::Connect::new(
            types::Fd(self.fd.as_raw_fd()),
            &raw const self.sock_addr.storage as _,
            self.sock_addr.len,
        )
        .build()
    }

    type Buf = Box<SockAddrRaw>;

    fn into_buf(self) -> Self::Buf {
        self.sock_addr
    }
}

impl<'fd> Connect<'fd> {
    pub fn new(fd: &'fd Socket, sock_addr: Box<SockAddrRaw>) -> Result<Self, Error> {
        Ok(Self { fd, sock_addr })
    }
}

pub struct Timeout {
    timespec: Box<Timespec>,
    count: u32,
    flags: TimeoutFlags,
}

impl OpCode for Timeout {
    type Output = ();

    fn output(self, entry: cqueue::Entry) -> Result<Self::Output, Error> {
        match entry.as_result() {
            Ok(_) => Ok(()), // TODO: not sure when this happens what we should do
            Err(Error::Io(err)) if err.raw_os_error() == Some(62) => Ok(()),
            Err(err) => Err(err),
        }
    }

    unsafe fn sqe(&mut self) -> squeue::Entry {
        opcode::Timeout::new(&*self.timespec as *const _)
            .count(self.count)
            .flags(self.flags)
            .build()
    }

    type Buf = Box<Timespec>;

    fn into_buf(self) -> Self::Buf {
        self.timespec
    }
}

impl Timeout {
    pub fn new(duration: Duration) -> Self {
        let timespec = Box::new(
            Timespec::new()
                .sec(duration.as_secs())
                .nsec(duration.subsec_nanos()),
        );
        Self {
            timespec,
            count: 0,
            flags: TimeoutFlags::empty(),
        }
    }

    pub fn count(mut self, count: u32) -> Self {
        self.count = count;
        self
    }

    pub fn flags(mut self, flags: TimeoutFlags) -> Self {
        self.flags = flags;
        self
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::test;
    use crate::uring::Completion;

    use super::*;

    #[test]
    fn test_nop() {
        test!(async {
            Completion::run(Nop).await.unwrap();
        })
    }

    #[test]
    fn test_timeout() {
        test!(async {
            let start = Instant::now();
            Completion::run(Timeout::new(Duration::from_millis(250)))
                .await
                .unwrap();
            let delta = start.elapsed().as_millis().abs_diff(250);
            assert!(delta <= 10, "Δt ({}) > 10ms", delta)
        })
    }
}
