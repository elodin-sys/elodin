use async_watcher::{notify::RecursiveMode, AsyncDebouncer};
use core::time::Duration;
use std::{io, path::PathBuf};
use tokio::task::JoinSet;

use tokio_util::sync::CancellationToken;

use crate::recipe::Recipe;

#[derive(Debug)]
pub struct Watcher {
    pub recipe: Recipe,
    timeout: Duration,
}

impl Watcher {
    pub fn new(recipe: Recipe) -> Self {
        Self {
            recipe,
            timeout: Duration::from_millis(200),
        }
    }

    pub async fn run(
        self,
        name: String,
        release: bool,
        cancel_token: CancellationToken,
        watch_dirs: impl Iterator<Item = PathBuf>,
    ) -> io::Result<()> {
        let (mut debouncer, mut file_events) = AsyncDebouncer::new_with_channel(self.timeout, None)
            .await
            .map_err(io::Error::other)?;
        for watch_dir in watch_dirs {
            debouncer
                .watcher()
                .watch(&watch_dir, RecursiveMode::Recursive)
                .map_err(io::Error::other)?;
        }
        let mut proc_cancel_token;
        loop {
            proc_cancel_token = cancel_token.child_token();
            let mut set = JoinSet::new();
            set.spawn(
                self.recipe
                    .clone()
                    .run(name.clone(), release, proc_cancel_token.clone()),
            );
            tokio::select! {
                _ = set.join_next() => {
                    break;
                }
                _ = cancel_token.cancelled() => {
                    break;
                }
                res = file_events.recv() => {
                    let Some(event) = res else {
                        break;
                    };
                    if let Err(errors) = event {
                        eprintln!("errors occured while watching dir {:?}", errors);
                    }
                    proc_cancel_token.cancel();
                    set.join_next().await;
                }
            }
        }

        Ok(())
    }
}
