use async_watcher::{AsyncDebouncer, notify::RecursiveMode};
use core::time::Duration;
use futures::Future;
use std::{io, path::PathBuf};
use stellarator::util::CancelToken;
use tokio::task::JoinSet;
use tracing::error;

use crate::error::Error;

/// Simulation outputs can live beside the script. Only source edits should
/// restart Python, not database writes or generated capture metadata.
pub fn python_source_changed(event: &async_watcher::notify::Event) -> bool {
    !event.kind.is_access()
        && event.paths.iter().any(|path| {
            path.extension().is_some_and(|extension| extension == "py")
                    // DB source snapshots are outputs too, including removal
                    // events when the next run replaces an existing recording.
                    && !path.ancestors().any(|ancestor| {
                        ancestor.ends_with("simulation_source/files")
                    })
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WatchExitPolicy {
    ReturnOnSuccess,
    WaitForChange,
}

pub async fn watch<F>(
    timeout: Duration,
    builder: impl Fn(CancelToken) -> F,
    cancel_token: CancelToken,
    dirs: impl Iterator<Item = PathBuf>,
    exit_policy: WatchExitPolicy,
) -> Result<(), Error>
where
    F: Future<Output = Result<(), Error>> + Send + Sync + 'static,
{
    watch_filtered(timeout, builder, cancel_token, dirs, exit_policy, |_| true).await
}

pub async fn watch_filtered<F>(
    timeout: Duration,
    builder: impl Fn(CancelToken) -> F,
    cancel_token: CancelToken,
    dirs: impl Iterator<Item = PathBuf>,
    exit_policy: WatchExitPolicy,
    accepts: impl Fn(&async_watcher::notify::Event) -> bool,
) -> Result<(), Error>
where
    F: Future<Output = Result<(), Error>> + Send + Sync + 'static,
{
    let (mut debouncer, mut file_events) = AsyncDebouncer::new_with_channel(timeout, None)
        .await
        .map_err(io::Error::other)?;
    let flat_map = dirs.flat_map(ignore::Walk::new);
    let files = flat_map;
    for res in files {
        let Ok(watch_dir) = res else {
            continue;
        };
        debouncer
            .watcher()
            .watch(watch_dir.path(), RecursiveMode::NonRecursive)
            .map_err(io::Error::other)?;
    }
    let mut proc_cancel_token;
    while !cancel_token.is_cancelled() {
        proc_cancel_token = cancel_token.child();
        let mut set = JoinSet::new();
        set.spawn(builder(proc_cancel_token.clone()));
        tokio::select! {
            _ = cancel_token.wait() => {
                set.join_next().await;
                break;
            }
            result = set.join_next() => {
                match result {
                    Some(Ok(Ok(())))
                        if exit_policy == WatchExitPolicy::ReturnOnSuccess =>
                    {
                        return Ok(());
                    }
                    Some(Ok(result)) => {
                        if let Err(err) = result {
                            error!(?err, "error running watched process");
                        }
                        tokio::select! {
                            _ = cancel_token.wait() => break,
                            res = next_change(&mut file_events, &accepts) => {
                                let Some(event) = res else {
                                    break;
                                };
                                if let Err(errors) = event {
                                    eprintln!("errors occurred while watching dir {:?}", errors);
                                }
                            }
                        }
                    }
                    Some(Err(_)) => return Err(Error::JoinError),
                    None => break,
                }
            }
            res = next_change(&mut file_events, &accepts) => {
                let Some(event) = res else {
                    break;
                };
                if let Err(errors) = event {
                    eprintln!("errors occurred while watching dir {:?}", errors);
                }
                proc_cancel_token.cancel();
                set.join_next().await;
            }
        }
    }

    Ok(())
}

type FileEvents = Result<Vec<async_watcher::DebouncedEvent>, Vec<async_watcher::notify::Error>>;

async fn next_change(
    events: &mut tokio::sync::mpsc::Receiver<FileEvents>,
    accepts: &impl Fn(&async_watcher::notify::Event) -> bool,
) -> Option<FileEvents> {
    while let Some(event) = events.recv().await {
        match &event {
            Ok(changes) if !changes.iter().any(|change| accepts(&change.event)) => continue,
            _ => return Some(event),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use stellarator::util::CancelToken;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn python_outputs_do_not_restart_but_source_edits_do() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let dir = std::env::temp_dir().join(format!("s10-source-watch-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("db")).unwrap();
        std::fs::create_dir_all(dir.join("db/simulation_source/files")).unwrap();
        std::fs::write(
            dir.join("db/simulation_source/files/sim.py"),
            "# old snapshot",
        )
        .unwrap();
        std::fs::write(dir.join("sim.py"), "# initial\n").unwrap();
        let runs = Arc::new(AtomicUsize::new(0));
        let cancel = CancelToken::new();
        let task = tokio::spawn({
            let dir = dir.clone();
            let runs = runs.clone();
            let cancel = cancel.clone();
            async move {
                watch_filtered(
                    Duration::from_millis(50),
                    |token| {
                        let dir = dir.clone();
                        let runs = runs.clone();
                        async move {
                            std::fs::write(dir.join("capture_view.json"), "{}").unwrap();
                            std::fs::write(dir.join("db/telemetry"), "frame").unwrap();
                            std::fs::remove_file(dir.join("db/simulation_source/files/sim.py"))
                                .unwrap();
                            std::fs::write(
                                dir.join("db/simulation_source/files/sim.py"),
                                "# snapshot",
                            )
                            .unwrap();
                            runs.fetch_add(1, Ordering::SeqCst);
                            token.wait().await;
                            Ok(())
                        }
                    },
                    cancel,
                    std::iter::once(dir.clone()),
                    WatchExitPolicy::ReturnOnSuccess,
                    python_source_changed,
                )
                .await
            }
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            while runs.load(Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
            assert_eq!(
                runs.load(Ordering::SeqCst),
                1,
                "generated files restarted Python"
            );
            std::fs::write(dir.join("sim.py"), "# edited\n").unwrap();
            while runs.load(Ordering::SeqCst) < 2 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("Python source edit did not restart the simulation");
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn python_filter_accepts_atomic_saves_and_deletions() {
        use async_watcher::notify::{
            Event, EventKind,
            event::{ModifyKind, RemoveKind, RenameMode},
        };
        let renamed = Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Both)))
            .add_path("sim.tmp".into())
            .add_path("sim.py".into());
        assert!(python_source_changed(&renamed));
        assert!(python_source_changed(
            &Event::new(EventKind::Remove(RemoveKind::File)).add_path("helper.py".into())
        ));
        assert!(!python_source_changed(
            &Event::new(EventKind::Any).add_path("__pycache__/sim.pyc".into())
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn returns_when_builder_completes_ok() {
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            watch(
                Duration::from_millis(50),
                |_| async { Ok(()) },
                CancelToken::new(),
                std::iter::empty(),
                WatchExitPolicy::ReturnOnSuccess,
            ),
        )
        .await;
        assert!(
            result.is_ok(),
            "watch hung after the watched process exited cleanly"
        );
        assert!(result.unwrap().is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn success_stays_in_watch_loop_when_waiting_for_change() {
        let cancel = CancelToken::new();
        let cancel_for_watch = cancel.clone();
        let handle = tokio::spawn(async move {
            watch(
                Duration::from_millis(50),
                |_| async { Ok(()) },
                cancel_for_watch,
                std::iter::empty(),
                WatchExitPolicy::WaitForChange,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(
            !handle.is_finished(),
            "watch returned after a successful sidecar exit"
        );
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("watch did not exit after cancel")
            .expect("watch task panicked");
        assert!(result.is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn error_stays_in_watch_loop_until_cancel() {
        let cancel = CancelToken::new();
        let cancel_for_watch = cancel.clone();
        let handle = tokio::spawn(async move {
            watch(
                Duration::from_millis(50),
                |_| async { Err(Error::JoinError) },
                cancel_for_watch,
                std::iter::empty(),
                WatchExitPolicy::ReturnOnSuccess,
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(
            !handle.is_finished(),
            "watch returned on builder error instead of waiting for reload/cancel"
        );
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("watch did not exit after cancel")
            .expect("watch task panicked");
        assert!(result.is_ok());
    }
}
