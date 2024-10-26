use std::{
    cell::RefCell,
    future::Future,
    io,
    pin::pin,
    task::{Context, Poll, Waker},
};

#[cfg(not(target_os = "linux"))]
pub mod poll;
#[cfg(target_os = "linux")]
pub mod uring;

#[cfg(not(target_os = "linux"))]
type DefaultReactor = poll::PollingReactor;
#[cfg(target_os = "linux")]
type DefaultReactor = uring::UringReactor;

pub mod buf;
pub mod fs;
pub mod net;
pub mod os;

mod noop_waker;

pub use maitake::task::JoinHandle;
#[cfg(not(target_os = "linux"))]
pub(crate) use poll as reactor;
#[cfg(target_os = "linux")]
pub(crate) use uring as reactor;

thread_local! {
    static EXEC: Executor = Executor::default();
}

#[derive(Debug)]
pub enum Error {
    SubmissionQueueFull,
    ExecutorAlreadyInstalled,
    Io(std::io::Error),
    InvalidSocketAddrType,
    InvalidPath,

    // ideally unreachable states
    CompletionStateMissing,
    PolledIgnoredCompletion,
    CompletionOpCodeMissing,
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<rustix::io::Errno> for Error {
    fn from(err: rustix::io::Errno) -> Self {
        Error::Io(io::Error::from_raw_os_error(err.raw_os_error()))
    }
}

pub struct Executor<R: Reactor = DefaultReactor> {
    reactor: RefCell<R>,
    scheduler: maitake::scheduler::LocalScheduler,
}

impl<R: Reactor> Executor<R> {
    pub fn run<O, F>(&self, func: impl FnOnce() -> F) -> Result<O, Error>
    where
        F: Future<Output = O> + 'static,
        O: 'static,
    {
        let future = func();
        let main_task = self.scheduler.spawn(future);
        let mut main_task = pin!(main_task);
        let waker = self.reactor.borrow_mut().waker();
        let mut cx = Context::from_waker(&waker);
        loop {
            self.reactor.borrow_mut().process_io()?;
            let tick = self.scheduler.tick();
            if let Poll::Ready(output) = main_task.as_mut().poll(&mut cx) {
                self.reactor.borrow_mut().finalize_io()?;
                return Ok(output.unwrap());
            }
            if !tick.has_remaining {
                self.reactor.borrow_mut().wait_for_io()?;
            }
        }
    }
}

pub trait Reactor {
    fn wait_for_io(&mut self) -> Result<(), Error>;
    fn process_io(&mut self) -> Result<(), Error>;
    fn finalize_io(&mut self) -> Result<(), Error>;
    fn waker(&self) -> Waker;
}

pub fn run<R, F>(func: impl FnOnce() -> F) -> R
where
    F: Future<Output = R> + 'static,
    R: 'static,
{
    Executor::with(|e| e.run(func).unwrap())
}

impl Executor<DefaultReactor> {
    pub(crate) fn with<R>(f: impl for<'a> FnOnce(&'a Self) -> R) -> R {
        EXEC.with(|exec| f(exec))
    }

    pub(crate) fn with_reactor<R>(f: impl for<'a> FnOnce(&'a mut DefaultReactor) -> R) -> R {
        EXEC.with(|exec| {
            let mut reactor = exec
                .reactor
                .try_borrow_mut()
                .expect("attempted nested reactor access");
            f(&mut reactor)
        })
    }
}

pub fn spawn<F>(f: F) -> JoinHandle<F::Output>
where
    F: Future + 'static,
    F::Output: Send + 'static,
{
    Executor::with(|exec| exec.scheduler.spawn(f))
}

#[macro_export]
macro_rules! test {
    ($fut:expr) => {
        std::thread::spawn(|| $crate::run(|| $fut))
            .join()
            .expect("join failed")
    };
}
