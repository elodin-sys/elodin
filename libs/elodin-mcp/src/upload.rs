//! `elodin upload`: stream a database as a sparse `.tar.zst` to an Elodin Cloud project in
//! parts. The archive is deterministic, so rerunning an interrupted upload regenerates the same
//! parts and skips the ones the server already holds.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail};
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::cloud::{Cloud, Project};

const IN_FLIGHT: usize = 4;
const RETRIES: u32 = 5;

/// What to upload: a database directory, or an archive holding exactly one.
pub enum Source {
    Dir(PathBuf),
    Archive { path: PathBuf, root: String },
}

impl Source {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        if path.is_dir() {
            if !path.join("db_state").exists() {
                bail!("{} is not an Elodin database (no db_state)", path.display());
            }
            return Ok(Source::Dir(path.to_path_buf()));
        }
        // Refuse anything beside the database, so companion directories never reach the cloud.
        let decoder = zstd::stream::read::Decoder::new(File::open(path)?)
            .with_context(|| format!("{} is not a zstd archive", path.display()))?;
        let mut archive = tar::Archive::new(decoder);
        let mut root: Option<String> = None;
        let mut has_state = false;
        for entry in archive.entries()? {
            let entry = entry?;
            let p = entry.path()?.into_owned();
            let mut parts = p
                .components()
                .filter(|c| !matches!(c, std::path::Component::CurDir));
            let first = parts
                .next()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .unwrap_or_default();
            match &root {
                None => root = Some(first.clone()),
                Some(r) if *r != first => {
                    bail!(
                        "{} holds more than the database directory ({first:?} beside {r:?})",
                        path.display()
                    )
                }
                _ => {}
            }
            has_state |=
                parts.next().is_some_and(|c| c.as_os_str() == "db_state") && parts.next().is_none();
        }
        match root {
            Some(root) if has_state => Ok(Source::Archive {
                path: path.to_path_buf(),
                root,
            }),
            _ => bail!(
                "{} holds no database directory with a db_state",
                path.display()
            ),
        }
    }

    pub fn name(&self) -> String {
        match self {
            Source::Dir(p) => p
                .canonicalize()
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .unwrap_or_default(),
            Source::Archive { root, .. } => root.clone(),
        }
    }

    /// An upper bound on the archive size, so the server can pick a part size.
    fn estimate(&self) -> u64 {
        match self {
            Source::Archive { path, .. } => std::fs::metadata(path).map_or(0, |m| m.len()),
            Source::Dir(dir) => allocated(dir),
        }
    }

    fn write_to(&self, out: impl Write) -> anyhow::Result<()> {
        match self {
            Source::Dir(dir) => elodin_db::archive::write(dir, out)?,
            Source::Archive { path, .. } => {
                let mut out = out;
                std::io::copy(&mut File::open(path)?, &mut out)?;
                out.flush()?;
            }
        }
        Ok(())
    }
}

fn allocated(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => allocated(&e.path()),
            Ok(_) => e.metadata().map_or(0, |m| file_bytes(&m)),
            Err(_) => 0,
        })
        .sum()
}

/// Allocated bytes: sparse AppendLogs are 8 GiB long but mostly holes.
fn file_bytes(m: &std::fs::Metadata) -> u64 {
    #[cfg(unix)]
    return std::os::unix::fs::MetadataExt::blocks(m) * 512;
    #[cfg(not(unix))]
    return m.len();
}

/// Cuts the archive stream into fixed-size parts for the uploader.
struct Parts {
    buf: Vec<u8>,
    size: usize,
    tx: tokio::sync::mpsc::Sender<Vec<u8>>,
}

impl Write for Parts {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(data);
        while self.buf.len() >= self.size {
            let rest = self.buf.split_off(self.size);
            let part = std::mem::replace(&mut self.buf, rest);
            self.tx
                .blocking_send(part)
                .map_err(|_| std::io::Error::from(std::io::ErrorKind::BrokenPipe))?;
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for Parts {
    fn drop(&mut self) {
        if !self.buf.is_empty() {
            let _ = self.tx.blocking_send(std::mem::take(&mut self.buf));
        }
    }
}

pub struct Options {
    pub name: Option<String>,
    pub tags: serde_json::Map<String, Value>,
    pub join: bool,
    pub pattern: Option<String>,
    pub wait: bool,
}

/// Upload `source` to `project`; returns the run as the API reports it.
pub async fn upload(
    cloud: &Cloud,
    project: &Project,
    source: Source,
    opts: Options,
) -> anyhow::Result<Value> {
    let name = opts.name.clone().unwrap_or_else(|| source.name());
    let body = json!({
        "name": name,
        "tags": opts.tags,
        "export": {"join": opts.join, "pattern": opts.pattern},
        "bytes": source.estimate(),
    });
    let (status, run) = cloud.create_run(project, body).await?;
    if status == reqwest::StatusCode::CONFLICT {
        bail!(
            "run {name:?} already exists in project {:?} ({}); pick another name with --name",
            project.name,
            run["state"].as_str().unwrap_or("?")
        );
    }
    if !status.is_success() {
        bail!(
            "API error ({status}): {}",
            run["error"].as_str().unwrap_or("request failed")
        );
    }
    let id = run["id"]
        .as_str()
        .context("the API returned no run id")?
        .to_string();
    let part_size = run["part_size"]
        .as_u64()
        .context("the API returned no part size")? as usize;
    let held: HashMap<u64, String> = run["parts"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|p| Some((p["part"].as_u64()?, p["sha256"].as_str()?.to_string())))
        .collect();
    if !held.is_empty() {
        eprintln!("Resuming {name}: the server holds {} parts", held.len());
    }

    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(2);
    let producer = std::thread::spawn(move || {
        let parts = Parts {
            buf: Vec::with_capacity(part_size),
            size: part_size,
            tx,
        };
        source.write_to(parts)
    });

    let mut whole = Sha256::new();
    let (mut n, mut total, mut sent) = (0usize, 0u64, 0usize);
    let mut inflight = FuturesUnordered::new();
    while let Some(part) = rx.recv().await {
        n += 1;
        total += part.len() as u64;
        whole.update(&part);
        let sha = hex::encode(Sha256::digest(&part));
        if held.get(&(n as u64)) == Some(&sha) {
            continue;
        }
        if inflight.len() >= IN_FLIGHT
            && let Some(r) = inflight.next().await
        {
            r?;
        }
        inflight.push(put_part(cloud, &id, n, part.into()));
        sent += 1;
        eprint!("\rUploading {name}: part {n} ({} MiB)   ", total >> 20);
    }
    while let Some(r) = inflight.next().await {
        r?;
    }
    producer
        .join()
        .map_err(|_| anyhow::anyhow!("archive writer panicked"))??;
    eprintln!(
        "\rUploaded {name}: {n} parts, {} MiB ({sent} sent)   ",
        total >> 20
    );
    let mut run = cloud
        .complete(&id, n, total, &hex::encode(whole.finalize()))
        .await?;

    if opts.wait {
        let mut last = String::new();
        loop {
            let state = run["state"].as_str().unwrap_or_default().to_string();
            if state != last {
                eprintln!("{name}: {state}");
                last = state.clone();
            }
            if state == "complete" || state == "failed" {
                break;
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
            run = cloud.run(&id).await?;
        }
    }
    Ok(run)
}

async fn put_part(cloud: &Cloud, run: &str, n: usize, data: bytes::Bytes) -> anyhow::Result<()> {
    let mut delay = Duration::from_millis(500);
    for attempt in 1..=RETRIES {
        match cloud.put_part(run, n, data.clone()).await {
            Ok(_) => return Ok(()),
            Err(e) if attempt == RETRIES => return Err(e.context(format!("upload part {n}"))),
            Err(_) => {
                tokio::time::sleep(delay).await;
                delay *= 2;
            }
        }
    }
    unreachable!()
}
