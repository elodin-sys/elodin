use async_watcher::{AsyncDebouncer, notify::RecursiveMode};
use core::time::Duration;
use futures::Future;
use std::{io, path::PathBuf};
use stellarator::util::CancelToken;
use tokio::task::JoinSet;

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
