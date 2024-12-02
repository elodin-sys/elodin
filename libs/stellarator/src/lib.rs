use std::{
    cell::RefCell,
    future::Future,
    pin::pin,
    task::{Context, Poll, Waker},
    time::Duration,
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
pub mod io;
pub mod net;
pub mod os;

mod noop_waker;

pub use maitake::task::JoinHandle;
use maitake::time::Timer;
#[cfg(not(target_os = "linux"))]
pub(crate) use poll as reactor;
#[cfg(target_os = "linux")]
pub(crate) use uring as reactor;

pub use maitake::sync;

thread_local! {
    static EXEC: Executor = Executor::default();
}

#[derive(Debug, thiserror::Error)]
#[cfg_attr(feature = "miette", derive(miette::Diagnostic))]
pub enum Error {
    #[error("submission queue full")]
    SubmissionQueueFull,
    #[error("executor already installed")]
    ExecutorAlreadyInstalled,
    #[error("io {0}")]
    Io(std::io::Error),
    #[error("invalid socket addr type")]
    InvalidSocketAddrType,
    #[error("invalid path")]
    InvalidPath,

    // ideally unreachable states
    #[error("completion state missing")]
    CompletionStateMissing,
    #[error("polled ignored completion")]
    PolledIgnoredCompletion,
    #[error("completion op code missing")]
    CompletionOpCodeMissing,

    #[error("buffer overflow")]
    BufferOverflow,
    #[error("end of file")]
    EOF,
    #[error("integer overflow")]
    IntegerOverflow,
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<rustix::io::Errno> for Error {
    fn from(err: rustix::io::Errno) -> Self {
        Error::Io(std::io::Error::from_raw_os_error(err.raw_os_error()))
    }
}

pub type BufResult<T, B> = (Result<T, Error>, B);

pub struct Executor<R: Reactor = DefaultReactor> {
    reactor: RefCell<R>,
    scheduler: maitake::scheduler::LocalScheduler,
    timer: maitake::time::Timer,
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
            self.timer.try_turn();
            self.reactor.borrow_mut().process_io()?;
            let tick = self.scheduler.tick();
            if let Poll::Ready(output) = main_task.as_mut().poll(&mut cx) {
                self.reactor.borrow_mut().finalize_io()?;
                return Ok(output.unwrap());
            }
            let turn = self.timer.try_turn();
            if !tick.has_remaining && turn.as_ref().map(|t| t.expired == 0).unwrap_or(true) {
                self.reactor
                    .borrow_mut()
                    .wait_for_io(turn.and_then(|turn| turn.time_to_next_deadline()))?;
            }
        }
    }
}

pub trait Reactor {
    fn wait_for_io(&mut self, timeout: Option<Duration>) -> Result<(), Error>;
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
pub fn sleep(duration: Duration) -> maitake::time::Sleep<'static> {
    let timer: &'static Timer = unsafe { &*Executor::with(|e| &e.timer as *const _) };
    timer.sleep(duration)
}

#[macro_export]
macro_rules! test {
    ($fut:expr) => {
        std::thread::spawn(|| $crate::run(|| $fut))
            .join()
            .expect("join failed")
    };
}

#[macro_export]
macro_rules! rent {
    ($call:expr, $buf:ident) => {{
        let (res, o) = $call;
        $buf = o;
        res
    }};
}

#[macro_export]
macro_rules! rent_read {
    ($call:expr, $buf:ident) => {
        $crate::rent!($call, $buf).map(|len| &$buf[..len])
    };
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_sleep() {
        test!(async {
            let start = Instant::now();
            println!("sleep start");
            sleep(Duration::from_millis(250)).await;
            println!("slept");
            let delta = start.elapsed().as_millis().abs_diff(250);
            assert!(delta <= 10, "Δt ({}) > 10ms", delta)
        })
    }
}
