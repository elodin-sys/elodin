use std::{
    future::Future,
    sync::{Arc, atomic::Ordering},
};

use stellarator::{
    struc_con::Joinable,
    util::{AtomicCell, CancelToken},
};

/// Tracks all work owned by one Impeller connection, including nested tasks.
#[derive(Clone)]
pub(crate) struct ConnectionTasks(Arc<Inner>);

struct Inner {
    cancel: CancelToken,
    active: AtomicCell<usize>,
}

struct TaskGuard(ConnectionTasks);

impl Drop for TaskGuard {
    fn drop(&mut self) {
        self.0.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

impl ConnectionTasks {
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

    pub(crate) fn spawn<F>(&self, future: F) -> stellarator::JoinHandle<()>
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

    pub(crate) fn stellar<F, Fut>(&self, f: F)
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

    pub(crate) fn query<F, Fut, T>(
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

    pub(crate) fn drive_ticks(&self, state: Arc<crate::FixedRateStreamState>) {
        let tasks = self.clone();
        self.spawn(async move {
            loop {
                if !state
                    .driver_spawned
                    .swap(true, std::sync::atomic::Ordering::SeqCst)
                {
                    struct DriverGuard(Arc<crate::FixedRateStreamState>);
                    impl Drop for DriverGuard {
                        fn drop(&mut self) {
                            self.0
                                .driver_spawned
                                .store(false, std::sync::atomic::Ordering::SeqCst);
                            self.0.driver_notify.wake_all();
                        }
                    }
                    let guard = DriverGuard(state.clone());
                    // Keep the clock independent of connection processing.
                    tasks.stellar(move || async move {
                        let _guard = guard;
                        crate::run_tick_driver(state).await;
                    });
                    return;
                }
                let _ = state
                    .driver_notify
                    .wait_for(|| {
                        !state
                            .driver_spawned
                            .load(std::sync::atomic::Ordering::SeqCst)
                    })
                    .await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DB, FixedRateStreamState, Timestamp};
    use std::time::Duration;

    async fn bounded<T>(future: impl Future<Output = T>) -> T {
        futures_lite::future::race(future, async {
            stellarator::sleep(Duration::from_secs(5)).await;
            panic!("connection cleanup timed out");
        })
        .await
    }

    #[stellarator::test]
    async fn cancellation_joins_a_pending_query_after_its_request_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(DB::create(dir.path().join("db")).unwrap());
        let weak = Arc::downgrade(&db);
        let started = Arc::new(AtomicCell::new(false));
        let tasks = ConnectionTasks::new();
        let query_started = started.clone();
        let query = tasks.query(move || async move {
            query_started.store(true);
            std::future::pending::<()>().await;
            drop(db);
        });
        bounded(started.wait_for(|started| started)).await;
        drop(query);
        bounded(tasks.finish()).await;
        assert!(weak.upgrade().is_none(), "query runtime retained the DB");
    }

    #[stellarator::test]
    async fn shared_tick_driver_moves_to_a_surviving_connection() {
        let state = Arc::new(FixedRateStreamState::new(
            1,
            Duration::from_millis(1),
            Timestamp(0),
            1000,
        ));
        let first = ConnectionTasks::new();
        let second = ConnectionTasks::new();
        first.drive_ticks(state.clone());
        bounded(
            state
                .tick_notify
                .wait_for(|| state.current_timestamp() != Timestamp(0)),
        )
        .await
        .unwrap();
        second.drive_ticks(state.clone());
        bounded(first.finish()).await;
        let tick = state.current_timestamp();
        bounded(
            state
                .tick_notify
                .wait_for(|| state.current_timestamp() != tick),
        )
        .await
        .unwrap();
        state.set_playing(false);
        bounded(second.finish()).await;
        assert!(!state.driver_spawned.load(Ordering::SeqCst));
        assert_eq!(Arc::strong_count(&state), 1);
    }
}
