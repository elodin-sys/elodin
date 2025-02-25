use crate::Error;
use crate::Executor;
use crate::Reactor;
use io_uring::{cqueue, squeue};
use maitake::scheduler::ExternalWaker as _;
use maitake::time::Timer;
use pin_project::{pin_project, pinned_drop};
use slab::Slab;
use std::any::Any;
use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;
use std::{cell::RefCell, task::Waker};
use waker_fn::waker_fn;

pub mod ops;

pub struct UringReactor {
    uring: SharedUring,
    states: Slab<OpState>,
}

impl UringReactor {
    pub fn submit_op<O: OpCode>(&mut self, mut op_code: O) -> Result<Completion<O>, (O, Error)> {
        let entry = self.states.vacant_entry();
        let sqe = unsafe { op_code.sqe().user_data(entry.key() as u64) };
        unsafe {
            if let Err(err) = self.uring.with_submission(|mut s| s.push(&sqe)) {
                return Err((op_code, Error::from(err)));
            }
        }
        let id = CompletionId(entry.key());

        entry.insert(OpState::Waiting(None));

        Ok(Completion {
            id,
            op_code: Some(op_code),
        })
    }

    pub fn poll_completion<O: OpCode>(
        &mut self,
        cx: &mut std::task::Context<'_>,
        completion: Pin<&mut Completion<O>>,
    ) -> Poll<O::Output> {
        let completion = completion.project();
        let Some(state) = self.states.get_mut(completion.id.0) else {
            let op_code = take_completion_op_code(completion);
            return Poll::Ready(op_code.output_from_error(Error::CompletionStateMissing));
        };

        match state {
            OpState::Waiting(Some(waker)) if waker.will_wake(cx.waker()) => {
                // nothing to do here
            }
            OpState::Waiting(_) => {
                *state = OpState::Waiting(Some(cx.waker().clone()));
            }
            OpState::Completed(_) => {
                let OpState::Completed(cqe) = self.states.remove(completion.id.0) else {
                    unreachable!()
                };
                let op_code = take_completion_op_code(completion);
                return Poll::Ready(op_code.output(cqe));
            }
            OpState::Ignored(_) => {
                let op_code = take_completion_op_code(completion);
                return Poll::Ready(op_code.output_from_error(Error::PolledIgnoredCompletion));
            }
        }
        Poll::Pending
    }

    pub fn drop_completion<O: OpCode>(&mut self, completion: Pin<&mut Completion<O>>) {
        let completion = completion.project();
        let Some(state) = self.states.get_mut(completion.id.0) else {
            return;
        };
        // SAFETY: We will be using `into_buf` later which we
        let Some(op_code) = (unsafe { completion.op_code.get_unchecked_mut().take() }) else {
            return;
        };
        let Some(value) =
            stack_dst::Value::<dyn Any, _>::new_stable(op_code.into_buf(), |p| p as _).ok()
        else {
            // TODO: handle gracefully
            return;
        };

        *state = OpState::Ignored(value)
    }
}

fn take_completion_op_code<O: OpCode>(completion: CompletionProj<'_, O>) -> O {
    // SAFETY: This is safe because we are using `Pin` to guarantee the location,
    // of the underlying buffer io_uring will be writing into. Since we have completed the operation,
    // we can now move that buffer safely.
    let op_code = unsafe { completion.op_code.get_unchecked_mut().take() };
    let Some(op_code) = op_code else {
        unreachable!(
            "op code already taken - this should never happen as we only take the op_code when we are returning `Poll::Ready`"
        )
    };
    op_code
}

impl Reactor for UringReactor {
    fn wait_for_io(&mut self, timeout: Option<Duration>) -> Result<(), Error> {
        if let Some(d) = timeout {
            let timespec = io_uring::types::Timespec::new()
                .sec(d.as_secs())
                .nsec(d.as_nanos() as u32);
            self.uring.with_submission(|mut s| {
                let sqe = io_uring::opcode::Timeout::new(&timespec)
                    .build()
                    .user_data(u64::MAX);
                unsafe { s.push(&sqe) }
            })?;
            self.uring.submitter().submit()?; // NOTE: not sure why this is required? It seems to be some sort of uring race condition?
        }
        // let mut args = io_uring::types::SubmitArgs::new();
        // let timeout = timeout.map(|d| {
        //     io_uring::types::Timespec::new()
        //         .sec(d.as_secs())
        //         .nsec(d.as_nanos() as u32)
        // });
        // if let Some(timeout) = timeout.as_ref() {
        //     args = args.timespec(timeout);
        // }
        // match self.uring.submitter().submit_with_args(1, &args) {
        // NOTE: the above code works on Linux 6.x but not 5.x kernels. We should switch to this when reasonable
        let res = self.uring.submitter().submit_and_wait(1);
        match res {
            Ok(_) => Ok(()),
            Err(err) if err.raw_os_error() == Some(62) => Ok(()),
            Err(err) => Err(Error::from(err)),
        }
    }
    fn process_io(&mut self) -> Result<(), Error> {
        self.uring.submitter().submit()?;
        self.uring.with_completion(|c| {
            for comp in c {
                let id = comp.user_data() as usize;
                let Some(state) = self.states.get_mut(id) else {
                    // The most common case where this is skipped is when we
                    // submit a dummy operation to the reactor to make it complete instantly
                    continue;
                };
                let prev_state = mem::replace(state, OpState::Completed(comp));
                match prev_state {
                    OpState::Waiting(Some(waker)) => waker.wake(),
                    OpState::Ignored(value) => {
                        self.states.remove(id);
                        std::mem::drop(value);
                    }
                    _ => {}
                }
                //}
            }
            Ok(())
        })
    }

    fn finalize_io(&mut self) -> Result<(), Error> {
        self.uring.submitter().submit_and_wait(self.states.len())?;
        Ok(())
    }

    fn waker(&self) -> Waker {
        let waker = self.external_waker();
        waker_fn(move || {
            waker.wake();
        })
    }

    fn external_waker(&self) -> impl maitake::scheduler::ExternalWaker {
        ExternalWaker {
            ring: self.uring.clone(),
        }
    }
}

pub trait OpCode {
    type Output;

    fn output(self, cqe: cqueue::Entry) -> Self::Output;

    fn output_from_error(self, err: Error) -> Self::Output;

    /// # Safety
    /// Implementors of `sqe` must ensure that the buffers passed to io_uring
    /// will be pinned in place for the duration of the operation
    unsafe fn sqe(&mut self) -> squeue::Entry;

    type Buf: 'static;
    fn into_buf(self) -> Self::Buf;
}

pub enum OpState {
    Waiting(Option<Waker>),
    Completed(cqueue::Entry),
    Ignored(stack_dst::Value<dyn core::any::Any, stack_dst::buffers::Ptr8>),
}

pub struct CompletionId(pub usize);

#[pin_project(PinnedDrop, project = CompletionProj)]
pub struct Completion<O: OpCode> {
    id: CompletionId,
    #[pin]
    op_code: Option<O>,
}

impl<O: OpCode> Completion<O> {
    pub fn submit(op_code: O) -> Result<Self, (O, Error)> {
        Executor::with_reactor(|reactor| reactor.submit_op(op_code))
    }

    pub async fn run(op_code: O) -> O::Output {
        match Self::submit(op_code) {
            Ok(comp) => comp.await,
            Err((op_code, err)) => op_code.output_from_error(err),
        }
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
        Executor::with_reactor(|r| {
            r.drop_completion(self);
        })
    }
}

impl From<io_uring::squeue::PushError> for Error {
    fn from(_: io_uring::squeue::PushError) -> Self {
        Error::SubmissionQueueFull
    }
}

impl Default for Executor<UringReactor> {
    fn default() -> Self {
        Self::try_new().unwrap()
    }
}

impl Executor<UringReactor> {
    pub fn try_new() -> Result<Self, Error> {
        let reactor = UringReactor {
            uring: SharedUring::with_uring(io_uring::IoUring::new(256)?),
            states: Slab::with_capacity(256),
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

#[derive(Clone, Debug)]
struct SharedUring {
    uring: Arc<SharedUringInner>,
}

impl SharedUring {
    fn with_uring(ring: io_uring::IoUring) -> Self {
        let uring = SharedUringInner {
            completion_lock: spin::Mutex::new(()),
            submission_lock: spin::Mutex::new(()),
            ring,
        };
        SharedUring {
            uring: Arc::new(uring),
        }
    }

    fn with_submission<R>(
        &self,
        func: impl FnOnce(io_uring::squeue::SubmissionQueue<'_, io_uring::squeue::Entry>) -> R,
    ) -> R {
        self.uring.with_submission(func)
    }

    fn with_completion<R>(
        &self,
        func: impl FnOnce(io_uring::cqueue::CompletionQueue<'_, io_uring::cqueue::Entry>) -> R,
    ) -> R {
        self.uring.with_completion(func)
    }

    fn submitter(&self) -> io_uring::Submitter<'_> {
        self.uring.submitter()
    }
}

struct SharedUringInner {
    submission_lock: spin::Mutex<()>,
    completion_lock: spin::Mutex<()>,
    ring: io_uring::IoUring,
}

impl std::fmt::Debug for SharedUringInner {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl SharedUringInner {
    fn with_submission<R>(
        &self,
        func: impl FnOnce(io_uring::squeue::SubmissionQueue<'_, io_uring::squeue::Entry>) -> R,
    ) -> R {
        let _lock = self.submission_lock.lock();
        let res = func(unsafe { self.ring.submission_shared() });
        drop(_lock);
        res
    }
    fn with_completion<R>(
        &self,
        func: impl FnOnce(io_uring::cqueue::CompletionQueue<'_, io_uring::cqueue::Entry>) -> R,
    ) -> R {
        let _lock = self.completion_lock.lock();
        let res = func(unsafe { self.ring.completion_shared() });
        drop(_lock);
        res
    }

    fn submitter(&self) -> io_uring::Submitter<'_> {
        self.ring.submitter()
    }
}

#[derive(Debug)]
struct ExternalWaker {
    ring: SharedUring,
}

impl maitake::scheduler::ExternalWaker for ExternalWaker {
    fn wake(&self) {
        self.ring.with_submission(|mut s| {
            let sqe = io_uring::opcode::Nop::new().build().user_data(u64::MAX);
            let _ = unsafe { s.push(&sqe) };
        });
        let _ = self.ring.submitter().submit();
    }
}
