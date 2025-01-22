use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
};

use maitake::sync::WaitQueue;

pub struct CancelTokenInner {
    cancelled: AtomicBool,
    wait_cell: WaitQueue,
}

impl Default for CancelTokenInner {
    fn default() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            wait_cell: WaitQueue::new(),
        }
    }
}

impl CancelTokenInner {
    pub async fn wait(&self) {
        let _ = self
            .wait_cell
            .wait_for(|| self.cancelled.load(Ordering::SeqCst))
            .await;
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.wait_cell.wake_all();
    }
}

#[derive(Clone)]
pub struct CancelToken {
    parent: Weak<CancelTokenInner>,
    this: Arc<CancelTokenInner>,
}

impl Default for CancelToken {
    fn default() -> Self {
        let this = Arc::new(CancelTokenInner::default());
        Self {
            parent: Weak::new(),
            this,
        }
    }
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.this.cancel();
    }

    pub async fn wait(&self) {
        if let Some(parent) = &self.parent.upgrade() {
            futures_lite::future::race(parent.wait(), self.this.wait()).await;
        } else {
            self.this.wait().await
        }
    }

    pub fn is_cancelled(&self) -> bool {
        if let Some(parent) = &self.parent.upgrade() {
            self.this.is_cancelled() || parent.is_cancelled()
        } else {
            self.this.is_cancelled()
        }
    }

    pub fn child(&self) -> Self {
        CancelToken {
            parent: Arc::downgrade(&self.this),
            this: Arc::new(CancelTokenInner::default()),
        }
    }

    pub fn drop_guard(self) -> CancelTokenDropGuard {
        CancelTokenDropGuard(self)
    }
}

impl Drop for CancelTokenInner {
    fn drop(&mut self) {
        self.cancel();
    }
}

pub struct CancelTokenDropGuard(CancelToken);
impl Drop for CancelTokenDropGuard {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

struct OneshotInner<T> {
    value: UnsafeCell<MaybeUninit<T>>,
    wait_cell: WaitQueue,
}

pub struct OneshotRx<T>(Arc<OneshotInner<T>>);

impl<T> OneshotRx<T> {
    pub async fn wait(self) -> Option<T> {
        self.0.wait_cell.wait().await.ok()?;
        unsafe {
            let uninit = std::mem::replace(&mut *self.0.value.get(), MaybeUninit::zeroed());
            Some(uninit.assume_init())
        }
    }
}

pub struct OneshotTx<T>(Arc<OneshotInner<T>>);

unsafe impl<T> Send for OneshotInner<T> {}
unsafe impl<T> Sync for OneshotInner<T> {}

impl<T> OneshotTx<T> {
    pub fn send(self, value: T) {
        unsafe {
            let slot = &mut *self.0.value.get();
            slot.write(value);
        }
        self.0.wait_cell.wake();
    }
}

pub fn oneshot<T>() -> (OneshotTx<T>, OneshotRx<T>) {
    let inner = Arc::new(OneshotInner {
        value: UnsafeCell::new(MaybeUninit::uninit()),
        wait_cell: WaitQueue::new(),
    });
    (OneshotTx(inner.clone()), OneshotRx(inner))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_cancel() {
        crate::test!(async {
            let token = CancelToken::default();
            assert_eq!(token.is_cancelled(), false);
            token.cancel();
            token.wait().await;
        })
    }

    #[test]
    fn test_cancel_parent() {
        crate::test!(async {
            let parent = CancelToken::default();
            let a = parent.child();
            let b = parent.child();
            assert_eq!(a.is_cancelled(), false);
            b.cancel();
            assert_eq!(parent.is_cancelled(), false);
            parent.cancel();
            assert_eq!(a.is_cancelled(), true);
        })
    }
}
