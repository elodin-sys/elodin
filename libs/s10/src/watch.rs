use async_watcher::{AsyncDebouncer, notify::RecursiveMode};
use core::time::Duration;
use futures::Future;
use std::{io, path::PathBuf};
use stellarator::util::CancelToken;
use tokio::task::JoinSet;
use tracing::error;

use crate::error::Error;

pub async fn watch<F>(
    timeout: Duration,
    builder: impl Fn(CancelToken) -> F,
    cancel_token: CancelToken,
    dirs: impl Iterator<Item = PathBuf>,
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
            // The watched process exited on its own (e.g. `interactive=False`
            // reached `max_ticks`). Propagate that so a sim-led group can tear
            // down forever-running sidecars. A failure stays in the watch loop
            // until a file change restarts, matching `elodin run`'s reload DX.
            result = set.join_next() => {
                match result {
                    Some(Ok(Ok(()))) => return Ok(()),
                    Some(Ok(Err(err))) => {
                        error!(?err, "error running watched process");
                        tokio::select! {
                            _ = cancel_token.wait() => break,
                            res = file_events.recv() => {
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
            res = file_events.recv() => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use stellarator::util::CancelToken;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn returns_when_builder_completes_ok() {
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            watch(
                Duration::from_millis(50),
                |_| async { Ok(()) },
                CancelToken::new(),
                std::iter::empty(),
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
    async fn error_stays_in_watch_loop_until_cancel() {
        let cancel = CancelToken::new();
        let cancel_for_watch = cancel.clone();
        let handle = tokio::spawn(async move {
            watch(
                Duration::from_millis(50),
                |_| async { Err(Error::JoinError) },
                cancel_for_watch,
                std::iter::empty(),
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
