use super::{OpCode, OpState};
use crate::os::{BorrowedHandle, OwnedHandle};
use crate::BufResult;
use crate::{
    buf::{deref, deref_mut, IoBuf, IoBufMut},
    net::SockAddrRaw,
    os::AsRawOsHandle,
    Error, Executor,
};
use blocking::Task;
use pin_project::pin_project;
use polling::Event;
use rustix::net::RecvFlags;
use rustix::net::SendFlags;
use socket2::Socket;
use std::ffi::CString;
use std::{
    future::Future,
    io::{self},
    path::PathBuf,
    task::Poll,
};

#[cfg(not(target_os = "windows"))]
use std::os::fd::{AsFd, AsRawFd, FromRawFd};

#[cfg(target_os = "windows")]
use std::os::windows::io::{AsRawHandle, AsSocket, FromRawSocket};

#[pin_project]
pub struct Read<'fd, T> {
    #[pin]
    state: ReadState<'fd, T>,
    offset: u64,
}

#[pin_project(project = ReadStateProj)]
enum ReadState<'a, T> {
    Blocking(#[pin] blocking::Task<BufResult<usize, T>>),
    NonBlocking { sock: &'a Socket, buf: Option<T> },
}

impl<'fd, T: IoBufMut> Read<'fd, T> {
    pub fn new(fd: BorrowedHandle<'fd>, mut buf: T, offset: Option<u64>) -> Self {
        Self {
            state: match fd {
                BorrowedHandle::Fd(fd) => {
                    #[cfg(target_os = "windows")]
                    {
                        let fd = fd.as_raw_handle() as usize;
                        ReadState::Blocking(unblock(move || {
                            let fd = unsafe {
                                std::os::windows::io::BorrowedHandle::borrow_raw(fd as _)
                            };
                            let res =
                                crate::os::pread(&fd, deref_mut(&mut buf), offset).map_err(|err| {
                                    if let Some(os_err) = err.raw_os_error() {
                                        io::Error::from_raw_os_error(os_err)
                                    } else {
                                        io::Error::new(io::ErrorKind::Other, "")
                                    }
                                });
                            (res.map_err(Error::from), buf)
                        }))
                    }

                    #[cfg(not(target_os = "windows"))]
                    {
                        let fd = fd.as_raw_fd();
                        ReadState::Blocking(unblock(move || {
                            let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) };
                            let res = if let Some(offset) = offset {
                                rustix::io::pread(fd, deref_mut(&mut buf), offset)
                                    .map_err(|err| io::Error::from_raw_os_error(err.raw_os_error()))
                            } else {
                                rustix::io::read(fd, deref_mut(&mut buf))
                                    .map_err(|err| io::Error::from_raw_os_error(err.raw_os_error()))
                            };
                            (res.map_err(Error::from), buf)
                        }))
                    }
                }
                BorrowedHandle::Socket(sock) => ReadState::NonBlocking {
                    sock,
                    buf: Some(buf),
                },
            },
            offset: 0,
        }
    }

    pub fn offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }
}

impl<T: IoBufMut> OpCode for Read<'_, T> {
    type Output = BufResult<usize, T>;

    fn initial_state(&self) -> OpState {
        match &self.state {
            ReadState::Blocking(_) => OpState::Ready,
            ReadState::NonBlocking { .. } => OpState::Waiting(None),
        }
    }

    fn event(&self) -> Option<polling::Event> {
        match &self.state {
            ReadState::Blocking(_) => None,
            ReadState::NonBlocking { sock, .. } => {
                Some(Event::readable(sock.as_raw_os_handle() as usize))
            }
        }
    }

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.project();
        let state = this.state.project();
        match state {
            ReadStateProj::Blocking(mut task) => task.as_mut().poll(cx),
            ReadStateProj::NonBlocking { sock, buf } => {
                let mut temp_buf = buf.take().expect("buffer missing from op");
                #[cfg(not(target_os = "windows"))]
                let fd = sock.as_fd();
                #[cfg(target_os = "windows")]
                let fd = sock.as_socket();
                match rustix::net::recv(fd, deref_mut(&mut temp_buf), RecvFlags::empty())
                    .map_err(|err| io::Error::from_raw_os_error(err.raw_os_error()))
                    .to_poll()
                {
                    Poll::Ready(Ok(len)) => Poll::Ready((Ok(len), temp_buf)),
                    Poll::Ready(Err(err)) => Poll::Ready((Err(err), temp_buf)),
                    Poll::Pending => {
                        *buf = Some(temp_buf);
                        Poll::Pending
                    }
                }
            }
        }
    }
}
#[pin_project]
pub struct Write<'fd, T> {
    #[pin]
    state: WriteState<'fd, T>,
}

#[pin_project(project = WriteStateProj)]
enum WriteState<'a, T> {
    Blocking(#[pin] blocking::Task<BufResult<usize, T>>),
    NonBlocking { sock: &'a Socket, buf: Option<T> },
}

impl<'fd, T: IoBuf> Write<'fd, T> {
    pub fn new(fd: BorrowedHandle<'fd>, buf: T, offset: Option<u64>) -> Self {
        Self {
            state: match fd {
                BorrowedHandle::Fd(fd) => {
                    #[cfg(target_os = "windows")]
                    {
                        let fd = fd.as_raw_handle() as usize;
                        WriteState::Blocking(unblock(move || {
                            let fd = unsafe {
                                std::os::windows::io::BorrowedHandle::borrow_raw(fd as _)
                            };
                            let res = crate::os::pwrite(&fd, crate::buf::deref(&buf), offset)
                                .map_err(Error::from);
                            (res, buf)
                        }))
                    }

                    #[cfg(not(target_os = "windows"))]
                    {
                        let fd = fd.as_raw_fd();
                        WriteState::Blocking(unblock(move || {
                            let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) };

                            let res = if let Some(offset) = offset {
                                rustix::io::pwrite(fd, crate::buf::deref(&buf), offset)
                                    .map_err(|err| io::Error::from_raw_os_error(err.raw_os_error()))
                            } else {
                                rustix::io::write(fd, crate::buf::deref(&buf))
                                    .map_err(|err| io::Error::from_raw_os_error(err.raw_os_error()))
                            };
                            (res.map_err(Error::from), buf)
                        }))
                    }
                }
                BorrowedHandle::Socket(sock) => WriteState::NonBlocking {
                    sock,
                    buf: Some(buf),
                },
            },
        }
    }
}

impl<T: IoBuf> OpCode for Write<'_, T> {
    type Output = BufResult<usize, T>;

    fn initial_state(&self) -> OpState {
        match &self.state {
            WriteState::Blocking(_) => OpState::Ready,
            WriteState::NonBlocking { .. } => OpState::Waiting(None),
        }
    }

    fn event(&self) -> Option<polling::Event> {
        match &self.state {
            WriteState::Blocking(_) => None,
            WriteState::NonBlocking { sock, .. } => {
                Some(Event::writable(sock.as_raw_os_handle() as usize))
            }
        }
    }

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.project();
        let state = this.state.project();
        match state {
            WriteStateProj::Blocking(mut task) => task.as_mut().poll(cx),
            WriteStateProj::NonBlocking { sock, buf } => {
                let temp_buf = buf.take().expect("buffer missing from op");
                #[cfg(not(target_os = "windows"))]
                let fd = sock.as_fd();
                #[cfg(target_os = "windows")]
                let fd = sock.as_socket();
                match rustix::net::send(fd, deref(&temp_buf), SendFlags::empty())
                    .map_err(|err| io::Error::from_raw_os_error(err.raw_os_error()))
                    .to_poll()
                {
                    Poll::Ready(Ok(len)) => Poll::Ready((Ok(len), temp_buf)),
                    Poll::Ready(Err(err)) => Poll::Ready((Err(err), temp_buf)),
                    Poll::Pending => {
                        *buf = Some(temp_buf);
                        Poll::Pending
                    }
                }
            }
        }
    }
}

fn unblock<T, F>(f: F) -> Task<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let poller = Executor::with_reactor(|r| r.poller.clone());
    blocking::unblock(move || {
        let output = f();
        for _ in 0..1024 {
            if poller.notify().is_ok() {
                break;
            }
        }
        output
    })
}

#[pin_project]
pub struct Open(#[pin] blocking::Task<Result<OwnedHandle, Error>>);

impl OpCode for Open {
    fn event(&self) -> Option<polling::Event> {
        None
    }

    type Output = Result<OwnedHandle, Error>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.project();
        this.0.poll(cx)
    }

    fn initial_state(&self) -> OpState {
        OpState::Ready
    }
}

impl Open {
    pub fn new(path: PathBuf, options: &crate::fs::OpenOptions) -> Result<Self, Error> {
        let flags = options.access_mode()? | options.creation_mode()? | options.custom_flags;
        let mode = options.mode as libc::c_uint;
        Ok(Open(unblock(move || {
            let path = path.to_str().ok_or(Error::InvalidPath)?;
            let path = CString::new(path).map_err(io::Error::from)?;
            let fd = unsafe { libc::open(path.as_ptr(), flags, mode) }.as_result()?;
            #[cfg(target_os = "windows")]
            let fd = unsafe { libc::get_osfhandle(fd as _) };
            let handle = unsafe { OwnedHandle::from_raw_fd(fd as _) };
            Ok(handle)
        })))
    }
}

pub struct Connect<'fd> {
    fd: &'fd Socket,
}

impl<'fd> Connect<'fd> {
    #[allow(clippy::boxed_local)]
    pub fn new(fd: &'fd Socket, addr: Box<SockAddrRaw>) -> Result<Self, Error> {
        if unsafe {
            libc::connect(
                fd.as_raw_os_handle() as _,
                &raw const addr.storage as *const _,
                addr.len,
            )
        } == -1
        {
            let err = io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EINPROGRESS)
                && err.kind() != io::ErrorKind::WouldBlock
            {
                return Err(Error::Io(err));
            }
        }

        Ok(Self { fd })
    }
}

impl OpCode for Connect<'_> {
    type Output = Result<(), Error>;

    fn event(&self) -> Option<polling::Event> {
        Some(Event::writable(self.fd.as_raw_os_handle() as usize))
    }

    fn poll(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let socket = self.fd;
        if let Ok(Some(err)) | Err(err) = socket.take_error() {
            return Poll::Ready(Err(Error::Io(err)));
        }

        match socket.peer_addr() {
            Ok(_) => Poll::Ready(Ok(())),
            Err(err)
                if err.kind() == io::ErrorKind::NotConnected
                    || err.raw_os_error() == Some(libc::EINPROGRESS) =>
            {
                Poll::Pending
            }
            Err(err) => Poll::Ready(Err(Error::Io(err))),
        }
    }
}

pub struct Accept<'fd> {
    fd: &'fd Socket,
    sock_addr: Option<Box<SockAddrRaw>>,
}

impl<'fd> Accept<'fd> {
    pub fn new(fd: &'fd Socket, sock_addr: Box<SockAddrRaw>) -> Self {
        Self {
            fd,
            sock_addr: Some(sock_addr),
        }
    }
}

impl OpCode for Accept<'_> {
    type Output = BufResult<Socket, Box<SockAddrRaw>>;

    fn event(&self) -> Option<polling::Event> {
        Some(Event::readable(self.fd.as_raw_os_handle() as usize))
    }

    fn poll(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        let mut sock_addr = this.sock_addr.take().expect("sock addr missing");
        #[cfg(target_os = "windows")]
        const ERR_CODE: usize = windows_sys::Win32::Networking::WinSock::INVALID_SOCKET;
        #[cfg(not(target_os = "windows"))]
        const ERR_CODE: i32 = -1;

        match unsafe {
            libc::accept(
                this.fd.as_raw_os_handle() as _,
                &raw mut sock_addr.storage as *mut _ as *mut libc::sockaddr,
                &mut sock_addr.len,
            )
        } {
            code if code == ERR_CODE => {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::WouldBlock {
                    this.sock_addr = Some(sock_addr);
                    Poll::Pending
                } else {
                    Poll::Ready((Err(Error::from(err)), sock_addr))
                }
            }
            #[cfg(target_os = "windows")]
            fd => Poll::Ready((Ok(unsafe { Socket::from_raw_socket(fd as _) }), sock_addr)),
            #[cfg(not(target_os = "windows"))]
            fd => Poll::Ready((Ok(unsafe { Socket::from_raw_fd(fd) }), sock_addr)),
        }
    }

    fn initial_state(&self) -> OpState {
        OpState::Waiting(None)
    }
}

pub trait LibcExt: Sized + Copy {
    fn is_err(&self) -> bool;
    fn as_result(self) -> Result<Self, io::Error> {
        if self.is_err() {
            Err(io::Error::last_os_error())
        } else {
            Ok(self)
        }
    }
}

pub trait ResultExt<T> {
    fn to_poll(self) -> Poll<Result<T, Error>>;
}

impl<T> ResultExt<T> for io::Result<T> {
    fn to_poll(self) -> Poll<Result<T, Error>> {
        match self {
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => Poll::Pending,
            Err(err) => Poll::Ready(Err(Error::Io(err))),
            Ok(res) => Poll::Ready(Ok(res)),
        }
    }
}

impl LibcExt for libc::c_int {
    fn is_err(&self) -> bool {
        *self < 0
    }
}

impl LibcExt for libc::ssize_t {
    fn is_err(&self) -> bool {
        *self < 0
    }
}
