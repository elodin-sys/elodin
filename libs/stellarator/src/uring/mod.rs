use crate::Error;
use crate::Executor;
use crate::Reactor;
use io_uring::{cqueue, squeue};
use pin_project::{pin_project, pinned_drop};
use slab::Slab;
use std::any::Any;
use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::task::Poll;
use std::{cell::RefCell, task::Waker};

pub mod ops;

pub struct UringReactor {
    uring: io_uring::IoUring,
    states: Slab<OpState>,
}

impl UringReactor {
    pub fn submit_op<O: OpCode>(&mut self, mut op_code: O) -> Result<Completion<O>, (O, Error)> {
        let entry = self.states.vacant_entry();
        unsafe {
            let sqe = op_code.sqe().user_data(entry.key() as u64);
            if let Err(err) = self.uring.submission().push(&sqe) {
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
                // replace waker with this contexts waker if it is `None` or the current waker won't wake this context
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
        unreachable!("op code already taken - this should never happen as we only take the op_code when we are returning `Poll::Ready`")
    };
    op_code
}

impl Reactor for UringReactor {
    fn wait_for_io(&mut self) -> Result<(), Error> {
        self.uring.submit_and_wait(1)?;
        Ok(())
    }
    fn process_io(&mut self) -> Result<(), Error> {
        self.uring.submit()?;
        for comp in self.uring.completion() {
            let id = comp.user_data() as usize;
            let Some(state) = self.states.get_mut(id) else {
                // skipping for safety
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
        }
        Ok(())
    }

    fn finalize_io(&mut self) -> Result<(), Error> {
        self.uring.submit_and_wait(self.states.len())?;
        Ok(())
    }

    fn waker(&self) -> Waker {
        crate::noop_waker::noop_waker()
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
            uring: io_uring::IoUring::new(256)?,
            states: Slab::with_capacity(256),
        };
        Ok(Executor {
            reactor: RefCell::new(reactor),
            scheduler: maitake::scheduler::LocalScheduler::new(),
        })
    }
}
