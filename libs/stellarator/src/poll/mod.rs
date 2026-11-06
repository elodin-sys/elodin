use maitake::time::Timer;
use pin_project::{pin_project, pinned_drop};
use polling::Poller;
use slab::Slab;
use smallvec::SmallVec;
use std::{
    cell::RefCell,
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Poll, Waker},
    time::Duration,
};

#[cfg(not(target_os = "windows"))]
use std::os::fd::RawFd;
#[cfg(target_os = "windows")]
use std::os::windows::io::RawHandle as RawFd;

pub mod ops;

use crate::{Error, Executor, Reactor};

pub struct PollingReactor {
    poller: Arc<Poller>,
    states: Slab<OpState>,
    /// The ops waiting on each descriptor since its registration last fired,
    /// with the readiness each waits for.
    fds: HashMap<RawFd, SmallVec<[(CompletionId, polling::Event); 4]>>,
    events: polling::Events,
}

pub enum OpState {
    Waiting(Option<Waker>),
    Ready,
}

impl Reactor for PollingReactor {
    fn wait_for_io(&mut self, timeout: Option<Duration>) -> Result<(), crate::Error> {
        self.poller.wait(&mut self.events, timeout)?;
        Ok(())
    }

    fn process_io(&mut self) -> Result<(), crate::Error> {
        if self.events.is_empty() {
            self.poller.wait(&mut self.events, Some(Duration::ZERO))?;
        }
        for event in self.events.iter() {
            let Some(waiters) = self.fds.get_mut(&(event.key as RawFd)) else {
                continue;
            };
            // The registration fired, so wake every waiter; any that still
            // cannot proceed register again when they are next polled.
            for (id, _) in waiters.drain(..) {
                let Some(state) = self.states.get_mut(id.0) else {
                    continue;
                };
                if let OpState::Waiting(Some(waker)) = state {
                    waker.wake_by_ref();
                }
                *state = OpState::Ready;
            }
        }
        // `Poller::wait` appends to `events`. Left full, it hands kevent zero
        // slots, which returns at once even with a timeout: the reactor spins.
        self.events.clear();
        Ok(())
    }

    type RemainingIo = ();
    fn drain_io(&mut self) -> Self::RemainingIo {}

    fn waker(&self) -> Waker {
        let poller = self.poller.clone();
        waker_fn::waker_fn(move || {
            for _ in 0..1024 {
                if poller.notify().is_ok() {
                    break;
                }
            }
        })
    }

    fn external_waker(&self) -> impl maitake::scheduler::ExternalWaker {
        ExternalWaker {
            poller: self.poller.clone(),
        }
    }
}
#[derive(Debug)]
struct ExternalWaker {
    poller: Arc<Poller>,
}

impl maitake::scheduler::ExternalWaker for ExternalWaker {
    fn wake(&self) {
        for _ in 0..1024 {
            if self.poller.notify().is_ok() {
                break;
            }
        }
    }
}

impl PollingReactor {
    pub fn submit_op<O: OpCode>(&mut self, op_code: O) -> Result<Completion<O>, Error> {
        let id = CompletionId(self.states.insert(op_code.initial_state()));
        self.submit_op_reactor(&op_code, id)?;
        Ok(Completion { id, op_code })
    }

    fn submit_op_reactor(&mut self, op_code: &impl OpCode, id: CompletionId) -> Result<(), Error> {
        if let Some(event) = op_code.event() {
            let fd = event.key as RawFd;
            #[cfg(not(target_os = "windows"))]
            let source = unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) };
            #[cfg(target_os = "windows")]
            let source = unsafe { std::os::windows::io::BorrowedSocket::borrow_raw(fd as _) };
            if let Some(waiters) = self.fds.get_mut(&fd) {
                match waiters.iter_mut().find(|(waiter, _)| waiter.0 == id.0) {
                    Some((_, interest)) => *interest = event,
                    None => waiters.push((id, event)),
                }
                // A descriptor has one registration, and each change replaces
                // it; it must cover every waiter, or registering a reader would
                // drop a waiting writer's interest.
                self.poller.modify(source, combined(event.key, waiters))?;
            } else {
                unsafe { self.poller.add(&source, event)? };
                self.fds.insert(fd, smallvec::smallvec![(id, event)]);
            }
        }
        Ok(())
    }

    pub fn poll_completion<O: OpCode>(
        &mut self,
        cx: &mut std::task::Context<'_>,
        completion: Pin<&mut Completion<O>>,
    ) -> Poll<O::Output> {
        let mut completion = completion.project();
        let state = self
            .states
            .get_mut(completion.id.0)
            .expect("state not found for completion");

        match state {
            OpState::Waiting(Some(waker)) if waker.will_wake(cx.waker()) => {
                // nothing to do here
            }
            OpState::Waiting(_) => {
                // replace waker with this contexts waker if it is `None` or the current waker won't wake this context
                *state = OpState::Waiting(Some(cx.waker().clone()));
            }
            OpState::Ready => {}
        }
        match completion.op_code.as_mut().poll(cx) {
            Poll::Pending => {
                let op_code = completion.op_code.as_ref();
                *state = OpState::Waiting(Some(cx.waker().clone()));
                self.submit_op_reactor(op_code.get_ref(), *completion.id)
                    .expect("failed to submit op to reactor");
                Poll::Pending
            }
            Poll::Ready(res) => {
                let _ = self.states.try_remove(completion.id.0);
                if let Some(event) = completion.op_code.as_ref().get_ref().event()
                    && let Some(waiters) = self.fds.get_mut(&(event.key as RawFd))
                {
                    waiters.retain(|(waiter, _)| waiter.0 != completion.id.0);
                }
                Poll::Ready(res)
            }
        }
    }
}

/// One registration covering the readiness any of `waiters` waits for.
fn combined(key: usize, waiters: &[(CompletionId, polling::Event)]) -> polling::Event {
    let mut interest = polling::Event::none(key);
    for (_, event) in waiters {
        interest.readable |= event.readable;
        interest.writable |= event.writable;
        interest.set_interrupt(interest.is_interrupt() || event.is_interrupt());
        interest.set_priority(interest.is_priority() || event.is_priority());
    }
    interest
}

pub trait OpCode {
    fn initial_state(&self) -> OpState {
        OpState::Ready
    }

    fn event(&self) -> Option<polling::Event> {
        None
    }

    type Output;
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output>;
}
#[derive(Clone, Copy)]
pub struct CompletionId(pub usize);

#[pin_project(PinnedDrop, project = CompletionProj)]
pub struct Completion<O: OpCode> {
    id: CompletionId,
    #[pin]
    op_code: O,
}

impl<O: OpCode> Completion<O> {
    pub fn submit(op_code: O) -> Result<Self, Error> {
        Executor::with_reactor(|reactor| reactor.submit_op(op_code))
    }

    pub async fn run(op_code: O) -> O::Output {
        Self::submit(op_code).expect("failed to submit op").await
    }
}

impl<O: OpCode> Future for Completion<O> {
    type Output = O::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        Executor::with_reactor(|r| r.poll_completion(cx, self))
    }
}

#[pinned_drop]
impl<O: OpCode> PinnedDrop for Completion<O> {
    fn drop(self: Pin<&mut Self>) {
        // TODO
        //Executor::with_reactor(|r| todo!())
    }
}

impl Default for Executor<PollingReactor> {
    fn default() -> Self {
        Self::try_new().unwrap()
    }
}

impl Executor<PollingReactor> {
    pub fn try_new() -> Result<Self, Error> {
        let reactor = PollingReactor {
            poller: Arc::new(Poller::new()?),
            states: Slab::with_capacity(256),
            events: Default::default(),
            fds: Default::default(),
        };
        let scheduler =
            maitake::scheduler::LocalScheduler::with_external_waker(reactor.external_waker());
        Ok(Executor {
            reactor: RefCell::new(reactor),
            scheduler,
            timer: Timer::new(crate::os::os_clock()),
        })
    }
}

#[cfg(all(test, not(target_os = "windows")))]
mod tests {
    use std::{future::Future, net::SocketAddr, pin::pin, rc::Rc, task::Poll, time::Duration};

    use rustix::time::{ClockId, clock_gettime};

    use crate::{
        Executor,
        net::{TcpListener, TcpStream, UdpSocket},
        test,
    };

    /// More than the poller's event buffer holds (1024 on macOS).
    const ROUND_TRIPS: usize = 2048;

    fn thread_cpu_time() -> Duration {
        let t = clock_gettime(ClockId::ThreadCPUTime);
        Duration::new(t.tv_sec as u64, t.tv_nsec as u32)
    }

    /// Awaits `fut`, counting the polls that returned `Pending`: each one
    /// waited for the reactor to deliver a readiness event.
    async fn counting_pending<F: Future>(fut: F, pending: &mut usize) -> F::Output {
        let mut fut = pin!(fut);
        std::future::poll_fn(|cx| {
            let poll = fut.as_mut().poll(cx);
            if poll.is_pending() {
                *pending += 1;
            }
            poll
        })
        .await
    }

    #[test]
    async fn idle_reactor_blocks_after_more_events_than_its_buffer_holds() {
        let a = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        let b = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        let a_addr = a.local_addr().unwrap();
        let b_addr = b.local_addr().unwrap();
        // The echo task replies only after `a` has started waiting, so every
        // receive on `a` is pending until the reactor delivers its event.
        let echo = crate::spawn(async move {
            let mut buf = vec![0u8; 16];
            for _ in 0..ROUND_TRIPS {
                let (n, out) = b.recv(buf).await;
                n.unwrap();
                b.send_to(b"y", a_addr).await.0.unwrap();
                buf = out;
            }
        });
        let mut pending = 0;
        let mut buf = vec![0u8; 16];
        for _ in 0..ROUND_TRIPS {
            a.send_to(b"x", b_addr).await.0.unwrap();
            let (n, out) = counting_pending(a.recv(buf), &mut pending).await;
            assert_eq!(n.unwrap(), 1);
            buf = out;
        }
        echo.await.unwrap();
        assert!(
            pending >= ROUND_TRIPS,
            "only {pending} receives waited on the reactor"
        );
        let before = thread_cpu_time();
        crate::sleep(Duration::from_millis(100)).await;
        let spent = thread_cpu_time() - before;
        assert!(
            spent < Duration::from_millis(20),
            "the reactor ran for {spent:?} of an idle 100 ms sleep"
        );
    }

    /// `fut`'s output, or `None` if `limit` passes first.
    async fn within<F: Future>(limit: Duration, fut: F) -> Option<F::Output> {
        let mut fut = pin!(fut);
        let mut timer = pin!(crate::sleep(limit));
        std::future::poll_fn(|cx| {
            if let Poll::Ready(out) = fut.as_mut().poll(cx) {
                return Poll::Ready(Some(out));
            }
            timer.as_mut().poll(cx).map(|()| None)
        })
        .await
    }

    #[test]
    async fn completed_ops_leave_no_waiters_registered() {
        let a = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        let b = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
        let a_addr = a.local_addr().unwrap();
        let b_addr = b.local_addr().unwrap();
        let echo = crate::spawn(async move {
            let mut buf = vec![0u8; 16];
            for _ in 0..ROUND_TRIPS {
                let (n, out) = b.recv(buf).await;
                n.unwrap();
                b.send_to(b"y", a_addr).await.0.unwrap();
                buf = out;
            }
        });
        let mut buf = vec![0u8; 16];
        for _ in 0..ROUND_TRIPS {
            a.send_to(b"x", b_addr).await.0.unwrap();
            let (n, out) = a.recv(buf).await;
            n.unwrap();
            buf = out;
        }
        echo.await.unwrap();
        // Each event walks its descriptor's waiters, so any left behind make
        // every later event slower.
        let registered: usize =
            Executor::with_reactor(|r| r.fds.values().map(|waiters| waiters.len()).sum());
        assert_eq!(
            registered, 0,
            "waiters left after {ROUND_TRIPS} round trips"
        );
    }

    #[test]
    async fn a_waiting_reader_does_not_cancel_a_waiting_writer() {
        const TOTAL: usize = 32 << 20;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = Rc::new(
            TcpStream::connect(listener.local_addr().unwrap())
                .await
                .unwrap(),
        );
        let server = listener.accept().await.unwrap();
        // Waits on `client` for the byte the server sends at the end.
        let reader = crate::spawn({
            let client = client.clone();
            async move { client.read(vec![0u8; 16]).await.0.unwrap() }
        });
        let drain = crate::spawn(async move {
            let (mut got, mut buf) = (0, vec![0u8; 64 << 10]);
            while got < TOTAL {
                let (n, out) = server.read(buf).await;
                got += n.unwrap();
                buf = out;
            }
            server.write(b"x").await.0.unwrap();
        });
        // More than the socket buffers hold: the writer waits for writability
        // while the reader waits for readability on the same descriptor.
        let write_all = async {
            let (mut sent, mut chunk) = (0, vec![0u8; 1 << 20]);
            while sent < TOTAL {
                let (n, out) = client.write(chunk).await;
                sent += n.unwrap();
                chunk = out;
            }
        };
        let limit = Duration::from_secs(10);
        assert!(
            within(limit, write_all).await.is_some(),
            "the writer stalled"
        );
        assert!(
            within(limit, drain).await.map(Result::unwrap).is_some(),
            "the drain stalled"
        );
        assert_eq!(within(limit, reader).await.map(Result::unwrap), Some(1));
    }
}
