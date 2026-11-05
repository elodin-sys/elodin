use std::{future::Future, sync::Arc};

use stellarator::struc_con::Joinable;

/// Controls background work created by an embedded connection, including nested tasks.
///
/// Managed owners share a lifetime scope across clones and register work before
/// starting it, including when the scope is already closing. The caller controls
/// cancellation and awaits cleanup before removing DB storage. Local futures run
/// on the current Stellarator executor and need not be Send.
/// The `()` implementation uses detached tasks, without managed shutdown.
pub trait TaskOwner: Clone + Send + Sync + 'static {
    /// Start and own a local task, ignoring its output.
    fn spawn<F: Future + 'static>(&self, future: F) -> stellarator::JoinHandle<()>;

    /// Start and own a worker running a Stellarator executor.
    /// Completion must include executor teardown, not only the root future.
    fn spawn_stellar<F, Fut>(&self, f: F)
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future + 'static;

    /// Start a Tokio worker immediately, before the returned future is polled.
    /// Return None on cancellation. Keep ownership through worker teardown even
    /// if the request drops the returned future before awaiting its result.
    fn spawn_tokio<F, Fut, T>(
        &self,
        f: F,
    ) -> impl Future<Output = Result<Option<T>, stellarator::Error>>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = T>,
        T: Send + Sync + 'static;
}

// Preserve the ordinary server's detached runtime behavior.
impl TaskOwner for () {
    fn spawn<F: Future + 'static>(&self, future: F) -> stellarator::JoinHandle<()> {
        stellarator::spawn(async move {
            let _ = future.await;
        })
    }

    fn spawn_stellar<F, Fut>(&self, f: F)
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future + 'static,
    {
        stellarator::struc_con::stellar(move || async move {
            let _ = f().await;
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
        stellarator::struc_con::tokio(move |_| async move { Some(f().await) }).join()
    }
}

pub(crate) fn drive_ticks<T: TaskOwner>(owner: &T, state: Arc<crate::FixedRateStreamState>) {
    let tasks = owner.clone();
    owner.spawn(async move {
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
                tasks.spawn_stellar(move || async move {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_task_owner::OwnedTasks;
    use crate::{DB, FixedRateStreamState, Timestamp};
    use std::sync::atomic::Ordering;
    use std::time::Duration;
    use stellarator::util::AtomicCell;

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
        let tasks = OwnedTasks::new();
        let query_started = started.clone();
        let query = tasks.spawn_tokio(move || async move {
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
        let first = OwnedTasks::new();
        let second = OwnedTasks::new();
        drive_ticks(&first, state.clone());
        bounded(
            state
                .tick_notify
                .wait_for(|| state.current_timestamp() != Timestamp(0)),
        )
        .await
        .unwrap();
        drive_ticks(&second, state.clone());
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

#[cfg(test)]
mod closing_owner_tests {
    use super::*;
    use crate::test_task_owner::OwnedTasks;
    use stellarator::util::AtomicCell;

    #[stellarator::test]
    async fn an_already_canceled_owner_drops_new_work_without_polling_it() {
        let owner = OwnedTasks::new();
        drop(owner.cancel_guard());
        let polled = Arc::new(AtomicCell::new(false));
        let observed = polled.clone();
        let life = Arc::new(());
        let weak = Arc::downgrade(&life);
        owner
            .spawn(async move {
                polled.store(true);
                drop(life);
            })
            .await
            .unwrap();
        owner.finish().await;
        assert!(!observed.latest());
        assert!(weak.upgrade().is_none());
    }
}
