use async_watcher::{notify::RecursiveMode, AsyncDebouncer};
use core::time::Duration;
use std::{io, path::PathBuf};

use tokio_util::sync::CancellationToken;

use crate::posix;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug)]
pub struct Watcher {
    #[cfg_attr(feature = "serde", serde(flatten))]
    process: posix::Process,
    watch_dir: Option<PathBuf>,
    #[cfg_attr(feature = "serde", serde(default = "default_timeout"))]
    timeout: Duration,
}

fn default_timeout() -> Duration {
    Duration::from_millis(500)
}

impl Watcher {
    pub async fn run(self, cancel_token: CancellationToken) -> io::Result<()> {
        let (mut debouncer, mut file_events) = AsyncDebouncer::new_with_channel(self.timeout, None)
            .await
            .map_err(io::Error::other)?;
        let watch_dir = self
            .watch_dir
            .or(self.process.cwd.clone())
            .unwrap_or(std::env::current_dir()?);
        debouncer
            .watcher()
            .watch(&watch_dir, RecursiveMode::Recursive)
            .map_err(io::Error::other)?;
        let mut proc_cancel_token;
        loop {
            proc_cancel_token = cancel_token.child_token();
            let tracker = tokio_util::task::task_tracker::TaskTracker::new();
            tracker.spawn(self.process.clone().run(proc_cancel_token.clone()));
            tokio::select! {
                _ = tracker.wait() => {
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
                    tracker.wait().await;
                }
            }
        }

        Ok(())
    }
}
