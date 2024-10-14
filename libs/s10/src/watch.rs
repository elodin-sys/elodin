use async_watcher::{notify::RecursiveMode, AsyncDebouncer};
use core::time::Duration;
use futures::Future;
use std::{io, path::PathBuf};
use tokio::task::JoinSet;

use tokio_util::sync::CancellationToken;

use crate::error::Error;

pub async fn watch<F>(
    timeout: Duration,
    builder: impl Fn(CancellationToken) -> F,
    cancel_token: CancellationToken,
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
        let watch_dir = res?;
        debouncer
            .watcher()
            .watch(watch_dir.path(), RecursiveMode::NonRecursive)
            .map_err(io::Error::other)?;
    }
    let mut proc_cancel_token;
    loop {
        proc_cancel_token = cancel_token.child_token();
        let mut set = JoinSet::new();
        set.spawn(builder(proc_cancel_token.clone()));
        tokio::select! {
            res = set.join_next() => {
                if let Some(res) = res {
                    return res.map_err(|_| Error::JoinError)?;
                }
            }
            _ = cancel_token.cancelled() => {
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
