use std::{
    future::Future,
    sync::{Arc, atomic::Ordering},
};

use crate::TaskOwner;

use stellarator::{
    struc_con::Joinable,
    util::{AtomicCell, CancelToken},
};

// A caller-side lifetime policy, outside the public DB implementation.
#[derive(Clone)]
pub(crate) struct OwnedTasks(Arc<Inner>);

struct Inner {
    cancel: CancelToken,
    active: AtomicCell<usize>,
}

struct TaskGuard(OwnedTasks);

impl Drop for TaskGuard {
    fn drop(&mut self) {
        self.0.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

impl OwnedTasks {
    pub(crate) fn new() -> Self {
        Self(Arc::new(Inner {
            cancel: CancelToken::new(),
            active: AtomicCell::new(0),
        }))
    }

    fn register(&self) -> TaskGuard {
        self.0.active.fetch_add(1, Ordering::SeqCst);
        TaskGuard(self.clone())
    }

    pub(crate) fn cancel(&self) {
        self.0.cancel.cancel();
    }

    pub(crate) fn cancel_guard(&self) -> stellarator::util::CancelTokenDropGuard {
        self.0.cancel.clone().drop_guard()
    }

    pub(crate) async fn finish(&self) {
        self.cancel();
        self.0.active.wait_for(|active| active == 0).await;
    }

    pub(crate) async fn run<F: Future>(&self, future: F) -> Option<F::Output> {
        // Poll cancellation first, including when a task starts after shutdown.
        futures_lite::future::or(
            async {
                self.0.cancel.wait().await;
                None
            },
            async { Some(future.await) },
        )
        .await
    }
}

impl TaskOwner for OwnedTasks {
    fn spawn<F>(&self, future: F) -> stellarator::JoinHandle<()>
    where
        F: Future + 'static,
    {
        let guard = self.register();
        let tasks = self.clone();
        stellarator::spawn(async move {
            let _guard = guard;
            let _ = tasks.run(future).await;
        })
    }

    fn spawn_stellar<F, Fut>(&self, f: F)
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future + 'static,
    {
        let guard = self.register();
        let tasks = self.clone();
        std::thread::spawn(move || {
            // Release the registration only after the executor drops its tasks.
            let _guard = guard;
            stellarator::run(move || async move {
                let _ = tasks.run(f()).await;
            });
        });
    }

    fn spawn_tokio<F, Fut, T>(
        &self,
        f: F,
    ) -> impl Future<Output = Result<Option<T>, stellarator::Error>>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = T>,
        T: Send + Sync + 'static,
    {
        let guard = self.register();
        let tasks = self.clone();
        let thread = stellarator::struc_con::tokio(move |_| async move { tasks.run(f()).await });
        let (tx, rx) = stellarator::util::oneshot();
        // This join task must outlive cancellation of the request future.
        stellarator::spawn(async move {
            let _guard = guard;
            tx.send(thread.join().await);
        });
        async move { rx.wait().await.ok_or(stellarator::Error::JoinFailed)? }
    }
}
