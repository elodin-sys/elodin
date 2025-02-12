use std::{
    cell::{RefCell, UnsafeCell},
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

pub use stellarator_buf as buf;
pub mod fs;
pub mod io;
pub mod net;
pub mod os;
pub mod struc_con;
pub mod util;

mod noop_waker;

use maitake::{scheduler::ExternalWaker, time::Timer};
use pin_project::pin_project;
#[cfg(not(target_os = "linux"))]
pub(crate) use poll as reactor;
#[cfg(target_os = "linux")]
pub(crate) use uring as reactor;

pub use maitake::sync;

thread_local! {
    static EXEC: UnsafeCell<Option<Executor>> = UnsafeCell::new(Some(Executor::default()));
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

    #[error("join failed")]
    JoinFailed,
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
        let out = loop {
            self.timer.try_turn();
            self.reactor.borrow_mut().process_io()?;
            let tick = self.scheduler.tick();
            if let Poll::Ready(output) = main_task.as_mut().poll(&mut cx) {
                self.reactor.borrow_mut().finalize_io()?;
                break Ok(output.unwrap());
            }
            let turn = self.timer.try_turn();
            if !tick.has_remaining && turn.as_ref().map(|t| t.expired == 0).unwrap_or(true) {
                self.reactor
                    .borrow_mut()
                    .wait_for_io(turn.and_then(|turn| turn.time_to_next_deadline()))?;
            }
        };
        unsafe {
            EXEC.with(|exec| {
                let exec = &mut *exec.get();
                drop(exec.take().expect("missing reactor"));
            })
        }
        out
    }
}

pub trait Reactor {
    fn wait_for_io(&mut self, timeout: Option<Duration>) -> Result<(), Error>;
    fn process_io(&mut self) -> Result<(), Error>;
    fn finalize_io(&mut self) -> Result<(), Error>;
    fn waker(&self) -> Waker;
    fn external_waker(&self) -> impl ExternalWaker;
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
        EXEC.with(|exec| {
            // safety: We only ever gain mutable access to the executor when `run` is complete,
            // and we are removing the executor so this is safe
            let exec = unsafe { &*exec.get() };
            f(exec.as_ref().expect("missing executor"))
        })
    }

    pub(crate) fn with_reactor<R>(f: impl for<'a> FnOnce(&'a mut DefaultReactor) -> R) -> R {
        Self::with(|exec| {
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
    JoinHandle(Executor::with(|exec| exec.scheduler.spawn(f)))
}

pub fn sleep(duration: Duration) -> maitake::time::Sleep<'static> {
    let timer: &'static Timer = unsafe { &*Executor::with(|e| &e.timer as *const _) };
    timer.sleep(duration)
}

#[pin_project]
pub struct JoinHandle<O>(#[pin] pub maitake::task::JoinHandle<O>);

impl<O> JoinHandle<O> {
    pub fn drop_guard(self) -> JoinHandleDropGuard<O> {
        JoinHandleDropGuard(self)
    }
}

pub struct JoinHandleDropGuard<T>(crate::JoinHandle<T>);

impl<T> Drop for JoinHandleDropGuard<T> {
    fn drop(&mut self) {
        self.0 .0.cancel();
    }
}

impl<O> Future for JoinHandle<O> {
    type Output = <maitake::task::JoinHandle<O> as Future>::Output;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.project().0.poll(cx)
    }
}

#[cfg(feature = "futures")]
impl<O> futures::future::FusedFuture for JoinHandle<O> {
    fn is_terminated(&self) -> bool {
        self.0.is_complete()
    }
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
    use std::{sync::Arc, time::Instant};

    use sync::WaitCell;

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

    #[test]
    fn test_cross_thread_wake() {
        let a = Arc::new(WaitCell::new());
        let b = a.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            b.wake();
            println!("woke wait cell");
        });
        test!(async move {
            println!("waiting wait cell");
            a.wait().await.unwrap();
        })
    }
}
