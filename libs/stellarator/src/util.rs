use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

use maitake::sync::WaitQueue;
pub use stellarator_buf::AtomicValue;

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

pub struct AtomicCell<A: AtomicValue> {
    pub value: A::Atomic,
    pub wait_queue: WaitQueue,
}

macro_rules! impl_atomic_numeric_ops {
    ($($t:ty),+ $(,)?) => {
        $(
            impl AtomicCell<$t> {
                pub fn fetch_add(&self, val: $t, order: Ordering) -> $t {
                    let prev = self.value.fetch_add(val, order);
                    self.wait_queue.wake_all();
                    prev
                }

                pub fn fetch_sub(&self, val: $t, order: Ordering) -> $t {
                    let prev = self.value.fetch_sub(val, order);
                    self.wait_queue.wake_all();
                    prev
                }

                pub fn fetch_and(&self, val: $t, order: Ordering) -> $t {
                    let prev = self.value.fetch_and(val, order);
                    self.wait_queue.wake_all();
                    prev
                }

                pub fn fetch_nand(&self, val: $t, order: Ordering) -> $t {
                    let prev = self.value.fetch_nand(val, order);
                    self.wait_queue.wake_all();
                    prev
                }

                pub fn fetch_or(&self, val: $t, order: Ordering) -> $t {
                    let prev = self.value.fetch_or(val, order);
                    self.wait_queue.wake_all();
                    prev
                }

                pub fn fetch_xor(&self, val: $t, order: Ordering) -> $t {
                    let prev = self.value.fetch_xor(val, order);
                    self.wait_queue.wake_all();
                    prev
                }

                pub fn fetch_max(&self, val: $t, order: Ordering) -> $t {
                    let prev = self.value.fetch_max(val, order);
                    self.wait_queue.wake_all();
                    prev
                }

                pub fn fetch_min(&self, val: $t, order: Ordering) -> $t {
                    let prev = self.value.fetch_min(val, order);
                    self.wait_queue.wake_all();
                    prev
                }
            }
        )+
    };
}

impl_atomic_numeric_ops! {i8, i16, i32, i64, isize, u8, u16, u32, u64, usize}

impl<A: AtomicValue> AtomicCell<A> {
    pub fn new(val: A) -> Self {
        AtomicCell {
            value: val.atomic(),
            wait_queue: WaitQueue::new(),
        }
    }

    pub fn store(&self, val: A) {
        A::store(&self.value, val, Ordering::Release);
        self.wait_queue.wake_all();
    }

    pub fn latest(&self) -> A {
        A::load(&self.value, Ordering::Acquire)
    }

    pub async fn wait(&self) {
        let _ = self.wait_queue.wait().await;
    }

    pub async fn wait_for(&self, f: impl Fn(A) -> bool) {
        let _ = self.wait_queue.wait_for(|| f(self.latest())).await;
    }

    pub async fn next(&self) -> A {
        let _ = self.wait().await;
        self.latest()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test;

    #[test]
    async fn test_simple_cancel() {
        let token = CancelToken::default();
        assert_eq!(token.is_cancelled(), false);
        token.cancel();
        token.wait().await;
    }

    #[test]
    async fn test_cancel_parent() {
        let parent = CancelToken::default();
        let a = parent.child();
        let b = parent.child();
        assert_eq!(a.is_cancelled(), false);
        b.cancel();
        assert_eq!(parent.is_cancelled(), false);
        parent.cancel();
        assert_eq!(a.is_cancelled(), true);
    }
}
