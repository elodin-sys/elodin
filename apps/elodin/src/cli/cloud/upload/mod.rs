//! Resumable zstd-compressed tar uploads of a stopped TelemetryDB, followed by publication polling.

use std::collections::HashSet;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Args;
use futures_util::{StreamExt, stream};
use miette::{Context, IntoDiagnostic};
use ring::digest::{Context as Digest, SHA256};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::runtime::Runtime;

use crate::cli::Cli;

use super::AuthCtx;
use super::auth::{random_token, tokens};

const PART_UPLOAD_CONCURRENCY: usize = 4;
const ARCHIVE_CACHE_VERSION: u32 = 1;
const ZSTD_LEVEL: i32 = 3;
const ZSTD_THREADS: u32 = 4;

#[derive(Args, Clone)]
pub struct UploadArgs {
    /// Stopped TelemetryDB directory containing db_state; compacted on upload without modifying it.
    pub db_path: PathBuf,
    /// Destination project UUID or name; optional when exactly one project is available.
    #[arg(long, env = "ELODIN_PROJECT")]
    pub project: Option<String>,
    /// Print the final published upload status as JSON.
    #[arg(long)]
    pub json: bool,
    /// Start a new upload instead of resuming the saved session for this snapshot.
    #[arg(long)]
    pub restart: bool,
    /// Maximum seconds to wait for publication after acceptance.
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..))]
    pub timeout: u64,
    /// Reuse prepared archives when source metadata is unchanged; must be outside the database.
    #[arg(long, env = "ELODIN_UPLOAD_CACHE_DIR")]
    pub archive_cache: Option<PathBuf>,
    /// Extra cache identity supplied by callers that pin this CLI.
    #[arg(
        long,
        env = "ELODIN_UPLOAD_CACHE_SALT",
        default_value = "",
        hide = true
    )]
    pub archive_cache_salt: String,
    /// Populate the prepared archive cache without contacting Elodin Cloud.
    #[arg(long, hide = true)]
    pub prepare_only: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct Manifest {
    version: u32,
    kind: String,
    files: Vec<UploadFile>,
}

#[derive(Debug, Deserialize, Serialize)]
struct UploadFile {
    part: String,
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Clone, Deserialize, Serialize)]
struct Archive {
    size_bytes: u64,
    sha256: String,
}

struct CheckedSource {
    file: std::io::Take<std::fs::File>,
    hash: Digest,
    bytes: u64,
}

struct CheckedOutput<W> {
    inner: W,
    hash: Digest,
    bytes: u64,
}

impl<W: Write> Write for CheckedOutput<W> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buffer)?;
        self.hash.update(&buffer[..written]);
        self.bytes += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl CheckedOutput<std::fs::File> {
    fn finish(self) -> std::io::Result<(u64, String)> {
        self.inner.sync_all()?;
        let sha256 = self
            .hash
            .finish()
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Ok((self.bytes, sha256))
    }
}

impl Read for CheckedSource {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let n = self.file.read(buffer)?;
        self.hash.update(&buffer[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
}

fn upload_file_len(path: &Path) -> miette::Result<u64> {
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(elodin_db::compact::is_append_log_file)
    {
        elodin_db::compact::compacted_file_len(path).into_diagnostic()
    } else {
        Ok(std::fs::metadata(path).into_diagnostic()?.len())
    }
}

#[cfg(test)]
fn archive(root: &Path, manifest: &Manifest) -> miette::Result<(tempfile::NamedTempFile, Archive)> {
    let output = tempfile::NamedTempFile::new().into_diagnostic()?;
    let checked = CheckedOutput {
        inner: output.reopen().into_diagnostic()?,
        hash: Digest::new(&SHA256),
        bytes: 0,
    };
    let mut encoder = zstd::stream::write::Encoder::new(checked, ZSTD_LEVEL).into_diagnostic()?;
    encoder.multithread(ZSTD_THREADS).into_diagnostic()?;
    encoder.include_checksum(true).into_diagnostic()?;
    let mut builder = tar::Builder::new(encoder);
    for entry in &manifest.files {
        let path = root.join(&entry.path);
        let file = std::fs::File::open(&path).into_diagnostic()?;
        if upload_file_len(&path)? != entry.size_bytes {
            return Err(miette::miette!(
                "upload source changed; keep the database stopped"
            ));
        }
        let mut source = CheckedSource {
            file: file.take(entry.size_bytes),
            hash: Digest::new(&SHA256),
            bytes: 0,
        };
        let name = ring::digest::digest(&SHA256, entry.part.as_bytes())
            .as_ref()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let mut header = tar::Header::new_gnu();
        header.set_size(entry.size_bytes);
        header.set_mode(0o600);
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(0);
        header.set_entry_type(tar::EntryType::Regular);
        builder
            .append_data(&mut header, name, &mut source)
            .into_diagnostic()?;
        if upload_file_len(&path)? != entry.size_bytes
            || source.bytes != entry.size_bytes
            || source
                .hash
                .finish()
                .as_ref()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
                != entry.sha256
        {
            return Err(miette::miette!(
                "upload source changed; keep the database stopped"
            ));
        }
    }
    let checked = builder
        .into_inner()
        .into_diagnostic()?
        .finish()
        .into_diagnostic()?;
    let (size_bytes, sha256) = checked.finish().into_diagnostic()?;
    Ok((output, Archive { size_bytes, sha256 }))
}

fn package(
    root: &Path,
    paths: Vec<(String, u64)>,
    output: tempfile::NamedTempFile,
) -> miette::Result<(Manifest, tempfile::NamedTempFile, Archive)> {
    let checked = CheckedOutput {
        inner: output.reopen().into_diagnostic()?,
        hash: Digest::new(&SHA256),
        bytes: 0,
    };
    let mut encoder = zstd::stream::write::Encoder::new(checked, ZSTD_LEVEL).into_diagnostic()?;
    encoder.multithread(ZSTD_THREADS).into_diagnostic()?;
    encoder.include_checksum(true).into_diagnostic()?;
    let mut builder = tar::Builder::new(encoder);
    let mut files = Vec::with_capacity(paths.len());
    for (path, size_bytes) in paths {
        let part = format!("file_{}", files.len());
        let disk_path = root.join(&path);
        let before = std::fs::metadata(&disk_path).into_diagnostic()?;
        let file = std::fs::File::open(&disk_path).into_diagnostic()?;
        let mut source = CheckedSource {
            file: file.take(size_bytes),
            hash: Digest::new(&SHA256),
            bytes: 0,
        };
        let name = ring::digest::digest(&SHA256, part.as_bytes())
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let mut header = tar::Header::new_gnu();
        header.set_size(size_bytes);
        header.set_mode(0o600);
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(0);
        header.set_entry_type(tar::EntryType::Regular);
        builder
            .append_data(&mut header, name, &mut source)
            .into_diagnostic()?;
        let sha256 = source
            .hash
            .finish()
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let after = std::fs::metadata(&disk_path).into_diagnostic()?;
        if upload_file_len(&disk_path)? != size_bytes
            || source.bytes != size_bytes
            || before.len() != after.len()
            || before.modified().ok() != after.modified().ok()
        {
            return Err(miette::miette!(
                "upload source changed; keep the database stopped"
            ));
        }
        files.push(UploadFile {
            part,
            path,
            size_bytes,
            sha256,
        });
    }
    let checked = builder
        .into_inner()
        .into_diagnostic()?
        .finish()
        .into_diagnostic()?;
    let (size_bytes, sha256) = checked.finish().into_diagnostic()?;
    Ok((
        Manifest {
            version: 1,
            kind: "telemetry_db".into(),
            files,
        },
        output,
        Archive { size_bytes, sha256 },
    ))
}

enum ArchiveFile {
    Temporary(tempfile::NamedTempFile),
    Cached(PathBuf),
}

struct CacheLock(PathBuf);

impl CacheLock {
    fn acquire(path: PathBuf) -> miette::Result<Self> {
        loop {
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let stale = std::fs::metadata(&path)
                        .and_then(|metadata| metadata.modified())
                        .and_then(|modified| modified.elapsed().map_err(std::io::Error::other))
                        .is_ok_and(|age| age > Duration::from_secs(7200));
                    if stale {
                        let _ = std::fs::remove_dir(&path);
                    } else {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
                Err(error) => return Err(error).into_diagnostic(),
            }
        }
    }
}

impl Drop for CacheLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.0);
    }
}

impl ArchiveFile {
    fn path(&self) -> &Path {
        match self {
            Self::Temporary(file) => file.path(),
            Self::Cached(path) => path,
        }
    }
}

#[derive(Deserialize, Serialize)]
struct CachedPackage {
    manifest: Manifest,
    archive: Archive,
}

#[cfg(test)]
fn prepare(
    root: &Path,
    cache: Option<&Path>,
) -> miette::Result<(Manifest, ArchiveFile, Archive, bool)> {
    prepare_salted(root, cache, "")
}

fn prepare_salted(
    root: &Path,
    cache: Option<&Path>,
    cache_salt: &str,
) -> miette::Result<(Manifest, ArchiveFile, Archive, bool)> {
    if let Some(cache) = cache {
        validate_cache_location(root, cache)?;
    }
    let paths = upload_paths(root)?;
    let Some(cache) = cache else {
        let output = tempfile::NamedTempFile::new().into_diagnostic()?;
        let (manifest, output, archive) = package(root, paths, output)?;
        return Ok((manifest, ArchiveFile::Temporary(output), archive, false));
    };
    std::fs::create_dir_all(cache).into_diagnostic()?;
    let key = archive_cache_key(root, &paths, cache_salt)?;
    let archive_path = cache.join(format!("{key}.tar.zst"));
    let metadata_path = cache.join(format!("{key}.json"));
    let _lock = CacheLock::acquire(cache.join(format!("{key}.lock")))?;
    if let Ok(bytes) = std::fs::read(&metadata_path)
        && let Ok(cached) = serde_json::from_slice::<CachedPackage>(&bytes)
        && std::fs::metadata(&archive_path)
            .is_ok_and(|metadata| metadata.len() == cached.archive.size_bytes)
    {
        return Ok((
            cached.manifest,
            ArchiveFile::Cached(archive_path),
            cached.archive,
            true,
        ));
    }
    let output = tempfile::NamedTempFile::new_in(cache).into_diagnostic()?;
    let (manifest, output, archive) = package(root, paths, output)?;
    let cached = CachedPackage { manifest, archive };
    remove_cached_file(&archive_path)?;
    output.persist(&archive_path).into_diagnostic()?;
    let mut metadata = tempfile::NamedTempFile::new_in(cache).into_diagnostic()?;
    metadata
        .write_all(&serde_json::to_vec(&cached).into_diagnostic()?)
        .into_diagnostic()?;
    metadata.as_file().sync_all().into_diagnostic()?;
    remove_cached_file(&metadata_path)?;
    metadata.persist(&metadata_path).into_diagnostic()?;
    let mut permissions = std::fs::metadata(&archive_path)
        .into_diagnostic()?
        .permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(&archive_path, permissions).into_diagnostic()?;
    Ok((
        cached.manifest,
        ArchiveFile::Cached(archive_path),
        cached.archive,
        false,
    ))
}

fn validate_cache_location(root: &Path, cache: &Path) -> miette::Result<()> {
    if cache
        .components()
        .any(|component| component == std::path::Component::ParentDir)
    {
        return Err(miette::miette!(
            "upload archive cache path cannot contain '..'"
        ));
    }
    let root = std::fs::canonicalize(root).into_diagnostic()?;
    let cache = prospective_path(cache)?;
    if cache.starts_with(&root) {
        return Err(miette::miette!(
            "upload archive cache cannot be inside the TelemetryDB"
        ));
    }
    Ok(())
}

fn prospective_path(path: &Path) -> miette::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir().into_diagnostic()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    let mut ancestor = normalized.as_path();
    while !ancestor.exists() {
        ancestor = ancestor
            .parent()
            .ok_or_else(|| miette::miette!("archive cache path has no existing ancestor"))?;
    }
    let resolved = std::fs::canonicalize(ancestor).into_diagnostic()?;
    Ok(resolved.join(normalized.strip_prefix(ancestor).into_diagnostic()?))
}

fn remove_cached_file(path: &Path) -> miette::Result<()> {
    #[cfg(windows)]
    if let Ok(metadata) = std::fs::metadata(path)
        && metadata.permissions().readonly()
    {
        let mut permissions = metadata.permissions();
        permissions.set_readonly(false);
        std::fs::set_permissions(path, permissions).into_diagnostic()?;
    }
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).into_diagnostic(),
    }
}

fn archive_cache_key(
    root: &Path,
    paths: &[(String, u64)],
    cache_salt: &str,
) -> miette::Result<String> {
    let mut hash = Digest::new(&SHA256);
    hash.update(
        format!("v{ARCHIVE_CACHE_VERSION}:zstd-{ZSTD_LEVEL}-threads-{ZSTD_THREADS}\0").as_bytes(),
    );
    hash.update(&(cache_salt.len() as u64).to_le_bytes());
    hash.update(cache_salt.as_bytes());
    let root = std::fs::canonicalize(root).into_diagnostic()?;
    let root_bytes = root.as_os_str().as_encoded_bytes();
    hash.update(&(root_bytes.len() as u64).to_le_bytes());
    hash.update(root_bytes);
    for (path, size) in paths {
        let metadata = std::fs::metadata(root.join(path)).into_diagnostic()?;
        let modified = metadata
            .modified()
            .into_diagnostic()?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        hash.update(&(path.len() as u64).to_le_bytes());
        hash.update(path.as_bytes());
        hash.update(&size.to_le_bytes());
        hash.update(&metadata.len().to_le_bytes());
        hash.update(&modified.to_le_bytes());
    }
    Ok(hash
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[derive(Deserialize)]
struct Project {
    id: uuid::Uuid,
    name: String,
}

#[derive(Deserialize)]
struct Projects {
    projects: Vec<Project>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Status {
    ticket: uuid::Uuid,
    project_id: uuid::Uuid,
    state: String,
    #[serde(default)]
    run_id: Option<uuid::Uuid>,
    #[serde(flatten)]
    details: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct FileSession {
    completed: bool,
    part_bytes: u64,
    part_count: u32,
    completed_parts: Vec<u32>,
    parts: Vec<PartUrl>,
}

#[derive(Clone, Deserialize)]
struct PartUrl {
    number: u32,
    offset: u64,
    size_bytes: u64,
    url: String,
}

#[derive(Serialize)]
struct PartReceipt {
    number: u32,
    etag: String,
}

#[derive(Default)]
struct TransferStats {
    parts: u32,
    retries: u32,
}

pub async fn run(ctx: &AuthCtx, args: &UploadArgs) -> miette::Result<()> {
    if args.prepare_only && args.archive_cache.is_none() {
        return Err(miette::miette!(
            "--prepare-only requires --archive-cache or ELODIN_UPLOAD_CACHE_DIR"
        ));
    }
    let root = args.db_path.clone();
    let cache = args.archive_cache.clone();
    let cache_salt = args.archive_cache_salt.clone();
    let prepare_started = Instant::now();
    let (manifest, archive_file, archive, cache_hit) =
        tokio::task::spawn_blocking(move || prepare_salted(&root, cache.as_deref(), &cache_salt))
            .await
            .into_diagnostic()??;
    eprintln!(
        "event=upload_performance stage=prepare duration_ms={} files={} archive_bytes={} cache_hit={} cache_version={} zstd_level={} zstd_threads={}",
        prepare_started.elapsed().as_millis(),
        manifest.files.len(),
        archive.size_bytes,
        cache_hit,
        ARCHIVE_CACHE_VERSION,
        ZSTD_LEVEL,
        ZSTD_THREADS
    );
    if args.prepare_only {
        println!("{}", archive_file.path().display());
        return Ok(());
    }
    let mut creds = tokens::ensure_valid(ctx).await?;
    let base = creds.api_url.trim_end_matches('/').to_owned();
    let projects: Projects = response(
        ctx.client
            .get(format!("{base}/v1/projects"))
            .bearer_auth(&creds.access_token)
            .send()
            .await
            .into_diagnostic()?,
    )
    .await?;
    let project = select_project(&projects.projects, args.project.as_deref())?;
    let url = format!("{base}/v1/projects/{project}/runs");
    let (session_path, idempotency) = upload_key(ctx, &creds, project, &manifest, args.restart)?;
    eprintln!(
        "Uploading {} files to project {project}...",
        manifest.files.len()
    );
    let resp = ctx
        .client
        .post(&url)
        .bearer_auth(&creds.access_token)
        .header("Idempotency-Key", &idempotency)
        .json(&manifest)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .into_diagnostic()
        .wrap_err("upload failed; keep the database stopped while uploading")?;
    let status_code = resp.status();
    let mut accepted: Status = response(resp).await?;
    if status_code != reqwest::StatusCode::ACCEPTED || accepted.project_id != project {
        return Err(miette::miette!("unexpected upload acceptance response"));
    }
    let ticket = accepted.ticket;
    if accepted.state == "uploading" {
        let transfer_started = Instant::now();
        let stats = transfer(ctx, &base, project, ticket, archive_file.path(), &archive).await?;
        eprintln!(
            "event=upload_performance stage=transfer duration_ms={} ticket={} archive_bytes={} parts={} retries={}",
            transfer_started.elapsed().as_millis(),
            ticket,
            archive.size_bytes,
            stats.parts,
            stats.retries
        );
        creds = tokens::ensure_valid(ctx).await?;
        accepted = response(
            ctx.client
                .post(format!(
                    "{base}/v1/projects/{project}/run-uploads/{ticket}/complete"
                ))
                .bearer_auth(&creds.access_token)
                .timeout(Duration::from_secs(30))
                .send()
                .await
                .into_diagnostic()?,
        )
        .await?;
        if accepted.ticket != ticket || accepted.project_id != project {
            return Err(miette::miette!("upload completion identity changed"));
        }
    }
    drop(archive_file);
    eprintln!("Upload accepted: {ticket}. Waiting for publication...");
    let status_url = format!(
        "{}/v1/projects/{project}/run-uploads/{ticket}",
        creds.api_url.trim_end_matches('/')
    );
    let wait = async {
        loop {
            creds = tokens::ensure_valid(ctx).await?;
            let status: Status = response(
                ctx.client
                    .get(&status_url)
                    .bearer_auth(&creds.access_token)
                    .timeout(Duration::from_secs(30))
                    .send()
                    .await
                    .into_diagnostic()?,
            )
            .await?;
            if status.ticket != ticket || status.project_id != project {
                return Err(miette::miette!("upload status identity changed"));
            }
            match status.state.as_str() {
                "succeeded" => {
                    let run = status
                        .run_id
                        .ok_or_else(|| miette::miette!("published upload omitted its run ID"))?;
                    std::fs::remove_file(&session_path).into_diagnostic()?;
                    if args.json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&status).into_diagnostic()?
                        );
                    } else {
                        println!("Uploaded TelemetryDB: run {run} (project {project})");
                    }
                    return Ok(());
                }
                "receiving" | "queued" | "running" | "retry_wait" => {}
                _ => {
                    return Err(miette::miette!(
                        "upload {ticket} did not publish: {} ({}); use --restart to start a new upload",
                        status.state,
                        serde_json::to_string(&status.details).into_diagnostic()?
                    ));
                }
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    };
    let publication_started = Instant::now();
    let result = tokio::time::timeout(Duration::from_secs(args.timeout), wait)
        .await
        .map_err(|_| {
            miette::miette!(
                "timed out waiting for upload {ticket}; it may still publish. Status: {status_url}"
            )
        })?;
    eprintln!(
        "event=upload_performance stage=publication_wait duration_ms={} ticket={}",
        publication_started.elapsed().as_millis(),
        ticket
    );
    result
}

async fn transfer(
    ctx: &AuthCtx,
    base: &str,
    project: uuid::Uuid,
    ticket: uuid::Uuid,
    archive_path: &Path,
    archive: &Archive,
) -> miette::Result<TransferStats> {
    // Never attach the API bearer token or follow redirects on storage requests.
    let storage_client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(180))
        .build()
        .into_diagnostic()?;
    let url = format!("{base}/v1/projects/{project}/run-uploads/{ticket}/archive");
    let mut stats = TransferStats::default();
    let mut failed_rounds = 0;
    let mut empty_refreshes = 0;
    loop {
        let session = archive_session(ctx, &url, archive).await?;
        let signed_at = Instant::now();
        if session.completed {
            return Ok(stats);
        }
        if session.parts.iter().any(|part| {
            let offset = u64::from(part.number - 1) * session.part_bytes;
            let size = (archive.size_bytes - offset).min(session.part_bytes);
            part.offset != offset || part.size_bytes != size
        }) {
            return Err(miette::miette!("invalid storage upload link"));
        }
        let mut failed = None;
        let mut refresh_urls = false;
        let parts_before = stats.parts;
        for batch in session.parts.chunks(8) {
            if signed_at.elapsed() >= Duration::from_secs(600) {
                refresh_urls = true;
                break;
            }
            let uploads = stream::iter(
                batch
                    .iter()
                    .cloned()
                    .map(|part| upload_part(&storage_client, base, archive_path, part)),
            )
            .buffer_unordered(PART_UPLOAD_CONCURRENCY);
            tokio::pin!(uploads);
            let mut receipts = Vec::with_capacity(batch.len());
            while let Some(result) = uploads.next().await {
                let uploaded = match result {
                    Ok(PartUpload::Uploaded(uploaded)) => uploaded,
                    Ok(PartUpload::Refresh) => {
                        refresh_urls = true;
                        continue;
                    }
                    Err(error) => {
                        failed.get_or_insert(error);
                        continue;
                    }
                };
                stats.parts += 1;
                stats.retries += uploaded.retries;
                receipts.push(uploaded.receipt);
            }
            acknowledge_parts(ctx, &url, &mut receipts).await?;
            if failed.is_some() || refresh_urls {
                break;
            }
        }
        if let Some(error) = failed {
            if stats.parts > parts_before {
                failed_rounds = 0;
            }
            failed_rounds += 1;
            if failed_rounds < 3 {
                tokio::time::sleep(Duration::from_secs(1 << (failed_rounds - 1))).await;
                continue;
            }
            return Err(error);
        }
        if refresh_urls {
            failed_rounds = 0;
            if stats.parts == parts_before {
                empty_refreshes += 1;
                if empty_refreshes >= 3 {
                    return Err(miette::miette!(
                        "storage upload links repeatedly expired without progress"
                    ));
                }
            } else {
                empty_refreshes = 0;
            }
            continue;
        }
        let creds = tokens::ensure_valid(ctx).await?;
        let resp = ctx
            .client
            .post(format!("{url}/complete"))
            .bearer_auth(&creds.access_token)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .into_diagnostic()?;
        expect_no_content(resp).await?;
        return Ok(stats);
    }
}

struct UploadedPart {
    receipt: PartReceipt,
    retries: u32,
}

enum PartUpload {
    Uploaded(UploadedPart),
    Refresh,
}

async fn upload_part(
    client: &reqwest::Client,
    base: &str,
    archive_path: &Path,
    part: PartUrl,
) -> miette::Result<PartUpload> {
    let storage_url = reqwest::Url::parse(&part.url).into_diagnostic()?;
    if (storage_url.scheme() != "https"
        && !(base.starts_with("http://") && storage_url.scheme() == "http"))
        || !storage_url.username().is_empty()
        || storage_url.password().is_some()
    {
        return Err(miette::miette!("invalid storage upload link"));
    }
    let mut file = tokio::fs::File::open(archive_path)
        .await
        .into_diagnostic()?;
    file.seek(std::io::SeekFrom::Start(part.offset))
        .await
        .into_diagnostic()?;
    let mut buffer = vec![0; part.size_bytes as usize];
    file.read_exact(&mut buffer)
        .await
        .into_diagnostic()
        .wrap_err("upload source changed; keep the database stopped")?;
    let bytes = bytes::Bytes::from(buffer);
    for retry in 0..3 {
        match client
            .put(storage_url.clone())
            .body(bytes.clone())
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                let etag = response
                    .headers()
                    .get("etag")
                    .and_then(|value| value.to_str().ok())
                    .ok_or_else(|| miette::miette!("storage omitted the part ETag"))?
                    .to_owned();
                return Ok(PartUpload::Uploaded(UploadedPart {
                    receipt: PartReceipt {
                        number: part.number,
                        etag,
                    },
                    retries: retry,
                }));
            }
            Ok(response)
                if matches!(
                    response.status(),
                    reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
                ) =>
            {
                return Ok(PartUpload::Refresh);
            }
            _ if retry < 2 => tokio::time::sleep(Duration::from_secs(1 << retry)).await,
            _ => break,
        }
    }
    Err(miette::miette!(
        "storage part upload failed; rerun the same command to resume"
    ))
}

async fn archive_session(
    ctx: &AuthCtx,
    url: &str,
    archive: &Archive,
) -> miette::Result<FileSession> {
    let creds = tokens::ensure_valid(ctx).await?;
    let session: FileSession = response(
        ctx.client
            .post(url)
            .bearer_auth(&creds.access_token)
            .json(archive)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .into_diagnostic()?,
    )
    .await?;
    let mut numbers = std::collections::HashSet::new();
    if session.part_bytes == 0
        || session.part_bytes > 64 * 1024 * 1024
        || u64::from(session.part_count) != archive.size_bytes.div_ceil(session.part_bytes)
        || session
            .completed_parts
            .iter()
            .any(|number| *number == 0 || *number > session.part_count || !numbers.insert(*number))
        || session.parts.iter().any(|part| {
            part.number == 0 || part.number > session.part_count || !numbers.insert(part.number)
        })
        || (!session.completed && numbers.len() != session.part_count as usize)
    {
        return Err(miette::miette!("invalid upload part layout"));
    }
    Ok(session)
}

async fn acknowledge_parts(
    ctx: &AuthCtx,
    url: &str,
    receipts: &mut Vec<PartReceipt>,
) -> miette::Result<()> {
    if receipts.is_empty() {
        return Ok(());
    }
    let creds = tokens::ensure_valid(ctx).await?;
    let resp = ctx
        .client
        .put(format!("{url}/parts"))
        .bearer_auth(&creds.access_token)
        .json(&serde_json::json!({"parts": receipts}))
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .into_diagnostic()?;
    expect_no_content(resp).await?;
    receipts.clear();
    Ok(())
}

async fn expect_no_content(resp: reqwest::Response) -> miette::Result<()> {
    if resp.status() == reqwest::StatusCode::NO_CONTENT {
        return Ok(());
    }
    let _: serde_json::Value = response(resp).await?;
    Err(miette::miette!("unexpected upload receipt response"))
}

fn upload_key(
    ctx: &AuthCtx,
    creds: &tokens::Credentials,
    project: uuid::Uuid,
    manifest: &Manifest,
    restart: bool,
) -> miette::Result<(PathBuf, String)> {
    use base64::Engine;
    use std::io::Write;
    #[derive(Deserialize, Serialize)]
    struct Session {
        key: String,
        created_at: u64,
    }
    // This claim separates local caches only. The API still performs authorization.
    let subject = creds
        .access_token
        .split('.')
        .nth(1)
        .and_then(|part| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(part)
                .ok()
        })
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|claims| {
            claims
                .get("sub")
                .and_then(|sub| sub.as_str())
                .map(str::to_owned)
        });
    let identity = serde_json::to_vec(&(&creds.issuer, &creds.api_url, subject, project, manifest))
        .into_diagnostic()?;
    let hash = ring::digest::digest(&SHA256, &identity)
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let directory = ctx.creds_path.with_file_name("upload-sessions");
    std::fs::create_dir_all(&directory).into_diagnostic()?;
    let path = directory.join(format!("{hash}.json"));
    if !restart {
        match std::fs::read(&path) {
            Ok(bytes) => {
                let saved: Session = serde_json::from_slice(&bytes)
                    .into_diagnostic()
                    .wrap_err("invalid saved upload session; use --restart")?;
                if saved.created_at.saturating_add(86_300) > tokens::now() {
                    return Ok((path, saved.key));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).into_diagnostic(),
        }
    }
    let session = Session {
        key: random_token(),
        created_at: tokens::now(),
    };
    let temporary = path.with_extension(format!("{}.tmp", random_token()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary).into_diagnostic()?;
    file.write_all(&serde_json::to_vec(&session).into_diagnostic()?)
        .into_diagnostic()?;
    file.sync_all().into_diagnostic()?;
    drop(file);
    std::fs::rename(&temporary, &path).into_diagnostic()?;
    Ok((path, session.key))
}

fn select_project(projects: &[Project], requested: Option<&str>) -> miette::Result<uuid::Uuid> {
    if let Some(value) = requested {
        // UUIDs are checked by the upload endpoint, including grants and tenant scope.
        if let Ok(id) = value.parse() {
            return Ok(id);
        }
        let matches: Vec<_> = projects.iter().filter(|p| p.name == value).collect();
        return match matches.as_slice() {
            [project] => Ok(project.id),
            _ => Err(miette::miette!(
                "project '{value}' not found or ambiguous; use its UUID"
            )),
        };
    }
    match projects {
        [] => Err(miette::miette!(
            "no projects available; run `elodin projects create <name>` first"
        )),
        [project] => Ok(project.id),
        _ if !std::io::stdin().is_terminal() => Err(miette::miette!(
            "multiple projects available; specify --project <name or UUID> or ELODIN_PROJECT"
        )),
        _ => {
            for (index, project) in projects.iter().enumerate() {
                eprintln!("  {}. {} ({})", index + 1, project.name, project.id);
            }
            eprintln!("Select a destination project by number:");
            let mut choice = String::new();
            std::io::stdin().read_line(&mut choice).into_diagnostic()?;
            choice
                .trim()
                .parse::<usize>()
                .ok()
                .and_then(|n| n.checked_sub(1))
                .and_then(|n| projects.get(n))
                .map(|p| p.id)
                .ok_or_else(|| miette::miette!("invalid project selection"))
        }
    }
}

async fn response<T: serde::de::DeserializeOwned>(resp: reqwest::Response) -> miette::Result<T> {
    let status = resp.status();
    if !status.is_success() {
        let detail: serde_json::Value = resp.json().await.unwrap_or_default();
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(miette::miette!(
                "the API rejected the token (401); run `elodin login` again"
            ));
        }
        return Err(miette::miette!("API error ({status}): {detail}"));
    }
    resp.json()
        .await
        .into_diagnostic()
        .wrap_err("invalid API response")
}

fn upload_paths(root: &Path) -> miette::Result<Vec<(String, u64)>> {
    if !std::fs::symlink_metadata(root).into_diagnostic()?.is_dir() {
        return Err(miette::miette!(
            "upload requires a TelemetryDB directory, not a file or symlink"
        ));
    }
    let mut paths = Vec::new();
    collect(root, root, &mut paths, 0)?;
    paths.sort();
    if !paths.iter().any(|path| path == Path::new("db_state")) {
        return Err(miette::miette!(
            "not a TelemetryDB directory: missing db_state"
        ));
    }
    let mut seen = HashSet::new();
    paths
        .into_iter()
        .map(|path| {
            let path = path
                .to_str()
                .ok_or_else(|| miette::miette!("non-UTF-8 upload path"))?
                .replace(std::path::MAIN_SEPARATOR, "/");
            if !portable_path(&path) || !seen.insert(path.to_ascii_lowercase()) {
                return Err(miette::miette!(
                    "nonportable or colliding upload path: {path}"
                ));
            }
            let compacted_len = upload_file_len(&root.join(&path))?;
            Ok((path, compacted_len))
        })
        .collect()
}

#[cfg(test)]
fn manifest(root: &Path) -> miette::Result<Manifest> {
    let paths = upload_paths(root)?;
    let mut files = Vec::with_capacity(paths.len());
    for (path, compacted_len) in paths {
        let disk_path = root.join(&path);
        let mut file = std::fs::File::open(&disk_path)
            .into_diagnostic()?
            .take(compacted_len);
        let mut digest = Digest::new(&SHA256);
        let mut size_bytes = 0u64;
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = file.read(&mut buf).into_diagnostic()?;
            if n == 0 {
                break;
            }
            digest.update(&buf[..n]);
            size_bytes = size_bytes
                .checked_add(n as u64)
                .ok_or_else(|| miette::miette!("file size overflow"))?;
        }
        if size_bytes != compacted_len {
            return Err(miette::miette!(
                "upload source changed while hashing: {path}"
            ));
        }
        let sha256 = digest
            .finish()
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        files.push(UploadFile {
            part: format!("file_{}", files.len()),
            path,
            size_bytes,
            sha256,
        });
    }
    Ok(Manifest {
        version: 1,
        kind: "telemetry_db".into(),
        files,
    })
}

fn collect(root: &Path, dir: &Path, paths: &mut Vec<PathBuf>, depth: usize) -> miette::Result<()> {
    if depth >= 32 {
        return Err(miette::miette!(
            "upload directory nesting exceeds 32 levels"
        ));
    }
    for entry in std::fs::read_dir(dir).into_diagnostic()? {
        let entry = entry.into_diagnostic()?;
        let kind = entry.file_type().into_diagnostic()?;
        if kind.is_dir() {
            collect(root, &entry.path(), paths, depth + 1)?;
        } else if kind.is_file() {
            paths.push(
                entry
                    .path()
                    .strip_prefix(root)
                    .into_diagnostic()?
                    .to_path_buf(),
            );
            if paths.len() > 20_000 {
                return Err(miette::miette!("upload exceeds 20000 files"));
            }
        } else {
            return Err(miette::miette!(
                "symlinks and special files cannot be uploaded: {}",
                entry.path().display()
            ));
        }
    }
    Ok(())
}

fn portable_path(path: &str) -> bool {
    path.len() <= 1024
        && path.split('/').all(|name| {
            let stem = name
                .split('.')
                .next()
                .unwrap_or_default()
                .trim_end_matches(' ')
                .to_ascii_uppercase();
            !name.is_empty()
                && name.len() <= 255
                && !matches!(name, "." | "..")
                && !name.ends_with(['.', ' '])
                && name.bytes().all(|b| {
                    (b.is_ascii_graphic()
                        && !matches!(b, b'\\' | b':' | b'<' | b'>' | b'"' | b'|' | b'?' | b'*'))
                        || b == b' '
                })
                && !matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
                && !(stem.len() == 4
                    && (stem.starts_with("COM") || stem.starts_with("LPT"))
                    && matches!(stem.as_bytes()[3], b'1'..=b'9'))
        })
}

impl Cli {
    pub fn upload(self, args: UploadArgs, rt: Runtime) -> miette::Result<()> {
        let ctx = self.auth_ctx()?;
        rt.block_on(run(&ctx, &args))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_concurrency_caps_part_memory() {
        assert!(PART_UPLOAD_CONCURRENCY * 64 * 1024 * 1024 <= 256 * 1024 * 1024);
    }

    #[test]
    fn archive_is_deterministic_compressed_and_detects_changed_sources() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("db_state"), b"abc").unwrap();
        std::fs::create_dir(dir.path().join("assets")).unwrap();
        std::fs::write(dir.path().join("assets/empty"), b"").unwrap();
        let manifest = manifest(dir.path()).unwrap();
        let (first, declaration) = archive(dir.path(), &manifest).unwrap();
        let (_, repeated) = archive(dir.path(), &manifest).unwrap();
        let (packaged, _, packaged_archive) = package(
            dir.path(),
            upload_paths(dir.path()).unwrap(),
            tempfile::NamedTempFile::new().unwrap(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&manifest).unwrap(),
            serde_json::to_value(packaged).unwrap()
        );
        assert_eq!(declaration.sha256, packaged_archive.sha256);
        assert_eq!(declaration.sha256, repeated.sha256);
        assert!(declaration.size_bytes < 2560);
        let decoder = zstd::stream::read::Decoder::new(first.reopen().unwrap()).unwrap();
        let mut tar = tar::Archive::new(decoder);
        let members = tar
            .entries()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(members.len(), 2);
        assert!(
            members
                .iter()
                .all(|entry| entry.header().entry_type().is_file())
        );
        std::fs::write(dir.path().join("db_state"), b"bad").unwrap();
        assert!(archive(dir.path(), &manifest).is_err());
    }

    #[test]
    fn prepared_archive_cache_reuses_unchanged_sources() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("db");
        let cache = dir.path().join("cache");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("db_state"), b"abc").unwrap();
        let (_, first, first_archive, first_hit) = prepare(&root, Some(&cache)).unwrap();
        let first_path = first.path().to_owned();
        assert!(!first_hit);
        assert!(
            std::fs::metadata(&first_path)
                .unwrap()
                .permissions()
                .readonly()
        );
        let (_, second, second_archive, second_hit) = prepare(&root, Some(&cache)).unwrap();
        assert!(second_hit);
        assert_eq!(first_path, second.path());
        assert_eq!(first_archive.sha256, second_archive.sha256);
        let sidecar = std::fs::read_dir(&cache)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "json")
            })
            .unwrap();
        std::fs::remove_file(sidecar).unwrap();
        let (_, rebuilt, _, rebuilt_hit) = prepare(&root, Some(&cache)).unwrap();
        assert!(!rebuilt_hit);
        assert_eq!(first_path, rebuilt.path());
        let (_, salted, _, salted_hit) =
            prepare_salted(&root, Some(&cache), "new-cli-revision").unwrap();
        assert!(!salted_hit);
        assert_ne!(first_path, salted.path());
        std::thread::sleep(Duration::from_millis(2));
        std::fs::write(root.join("db_state"), b"xyz").unwrap();
        let (_, changed, changed_archive, changed_hit) = prepare(&root, Some(&cache)).unwrap();
        assert!(!changed_hit);
        assert_ne!(first_path, changed.path());
        assert_ne!(first_archive.sha256, changed_archive.sha256);
    }

    #[test]
    fn prepared_archive_cache_rejects_database_subdirectories() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("db");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("db_state"), b"abc").unwrap();
        let cache = root.join(".cache/nested/archive");
        let error = prepare(&root, Some(&cache)).err().unwrap().to_string();
        assert!(error.contains("cannot be inside"));
        assert!(!root.join(".cache").exists());
    }

    #[cfg(unix)]
    #[test]
    fn prepared_archive_cache_rejects_symlink_parent_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("db");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(root.join("nested")).unwrap();
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(root.join("db_state"), b"abc").unwrap();
        std::os::unix::fs::symlink(root.join("nested"), outside.join("db-link")).unwrap();
        let cache = outside.join("db-link/../archive-cache");
        let error = prepare(&root, Some(&cache)).err().unwrap().to_string();
        assert!(error.contains("cannot contain '..'"));
        assert!(!dir.path().join("archive-cache").exists());
        assert!(!root.join("archive-cache").exists());
    }

    #[test]
    fn prepared_archive_cache_serializes_concurrent_builds() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("db");
        let cache = dir.path().join("cache");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("db_state"), b"abc").unwrap();
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let root = root.clone();
                let cache = cache.clone();
                std::thread::spawn(move || {
                    let (_, archive, _, hit) = prepare(&root, Some(&cache)).unwrap();
                    (archive.path().to_owned(), hit)
                })
            })
            .collect();
        let results: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();
        assert_eq!(results[0].0, results[1].0);
        assert_eq!(results.iter().filter(|(_, hit)| *hit).count(), 1);
    }

    #[tokio::test]
    async fn prepare_only_needs_no_cloud_session() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("db");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("db_state"), b"abc").unwrap();
        let ctx = AuthCtx {
            client: reqwest::Client::new(),
            issuer: "https://auth.invalid".into(),
            api_url: "https://api.invalid".into(),
            creds_path: dir.path().join("credentials.json"),
        };
        run(
            &ctx,
            &UploadArgs {
                db_path: root,
                project: None,
                json: false,
                restart: false,
                timeout: 1,
                archive_cache: Some(dir.path().join("cache")),
                archive_cache_salt: String::new(),
                prepare_only: true,
            },
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn prepare_only_requires_persistent_cache() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("db");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("db_state"), b"abc").unwrap();
        let ctx = AuthCtx {
            client: reqwest::Client::new(),
            issuer: "https://auth.invalid".into(),
            api_url: "https://api.invalid".into(),
            creds_path: dir.path().join("credentials.json"),
        };
        let error = run(
            &ctx,
            &UploadArgs {
                db_path: root,
                project: None,
                json: false,
                restart: false,
                timeout: 1,
                archive_cache: None,
                archive_cache_salt: String::new(),
                prepare_only: true,
            },
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains("requires --archive-cache"));
    }

    #[test]
    fn saved_upload_keys_resume_and_restart_without_touching_the_recording() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("db");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("db_state"), b"abc").unwrap();
        let manifest = manifest(&root).unwrap();
        let ctx = AuthCtx {
            client: reqwest::Client::new(),
            issuer: "https://auth.example.com".into(),
            api_url: "https://api.example.com".into(),
            creds_path: dir.path().join("credentials.json"),
        };
        let creds = tokens::Credentials {
            access_token: "test-token".into(),
            refresh_token: None,
            expires_at: tokens::now() + 1000,
            issuer: ctx.issuer.clone(),
            api_url: ctx.api_url.clone(),
        };
        let project = uuid::Uuid::nil();
        let key = upload_key(&ctx, &creds, project, &manifest, false).unwrap();
        assert_eq!(
            upload_key(&ctx, &creds, project, &manifest, false).unwrap(),
            key
        );
        let restarted = upload_key(&ctx, &creds, project, &manifest, true).unwrap();
        assert_ne!(restarted, key);
        assert_eq!(
            upload_key(&ctx, &creds, project, &manifest, false).unwrap(),
            restarted
        );
        assert_ne!(
            upload_key(&ctx, &creds, uuid::Uuid::new_v4(), &manifest, false).unwrap(),
            restarted
        );
        assert_eq!(std::fs::read(root.join("db_state")).unwrap(), b"abc");
        std::fs::remove_file(&restarted.0).unwrap();
        assert_ne!(
            upload_key(&ctx, &creds, project, &manifest, false).unwrap(),
            restarted
        );
        assert_eq!(std::fs::read_dir(root).unwrap().count(), 1);
    }

    #[test]
    fn manifest_compacts_sparse_logs_without_changing_source() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("db_state"), b"state").unwrap();
        let path = dir.path().join("data");
        let mut bytes = Vec::from(27u64.to_le_bytes());
        bytes.extend_from_slice(&[0; 16]);
        bytes.extend_from_slice(b"abc");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        file.set_len(8 * 1024 * 1024 * 1024).unwrap();
        let mut asset = Vec::from(16u64.to_le_bytes());
        asset.extend_from_slice(&[0; 24]);
        std::fs::write(dir.path().join("asset.bin"), &asset).unwrap();
        let manifest = manifest(dir.path()).unwrap();
        let asset_entry = manifest
            .files
            .iter()
            .find(|file| file.path == "asset.bin")
            .unwrap();
        assert_eq!(asset_entry.size_bytes, asset.len() as u64);
        archive(dir.path(), &manifest).unwrap();
        let entry = manifest
            .files
            .iter()
            .find(|file| file.path == "data")
            .unwrap();
        assert_eq!(entry.size_bytes, bytes.len() as u64);
        let expected: String = ring::digest::digest(&SHA256, &bytes)
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(entry.sha256, expected);
        assert_eq!(
            std::fs::metadata(path).unwrap().len(),
            8 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn manifest_is_sorted_and_hashes_exact_bytes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("db_state"), b"abc").unwrap();
        std::fs::create_dir(dir.path().join("assets")).unwrap();
        std::fs::write(dir.path().join("assets/empty"), b"").unwrap();
        let manifest = manifest(dir.path()).unwrap();
        assert_eq!(manifest.files[0].path, "assets/empty");
        assert_eq!(manifest.files[0].size_bytes, 0);
        assert_eq!(manifest.files[1].size_bytes, 3);
        assert_eq!(
            manifest.files[1].sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let json = serde_json::to_value(manifest).unwrap();
        assert_eq!(json["kind"], "telemetry_db");
        assert!(json.get("origin").is_none());
    }

    #[test]
    fn rejects_non_db_and_nonportable_paths() {
        let dir = tempfile::tempdir().unwrap();
        assert!(manifest(dir.path()).is_err());
        for path in ["../db_state", "a\\b", "a:b", "CON.txt", "a/COM1", "a.", "ü"] {
            assert!(!portable_path(path), "{path}");
        }
        assert!(portable_path("assets/schematics/main.kdl"));
        std::fs::write(dir.path().join("db_state"), b"").unwrap();
        std::fs::write(dir.path().join("DB_STATE"), b"").unwrap();
        // Case-insensitive filesystems fold these to one file.
        if std::fs::read_dir(dir.path()).unwrap().count() == 2 {
            assert!(manifest(dir.path()).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("db_state"), b"").unwrap();
        std::os::unix::fs::symlink("db_state", dir.path().join("link")).unwrap();
        assert!(manifest(dir.path()).is_err());
    }

    #[test]
    fn resolves_project_name_id_and_single_project() {
        let project = Project {
            id: uuid::Uuid::nil(),
            name: "demo".into(),
        };
        let projects = [project];
        assert_eq!(select_project(&projects, None).unwrap(), uuid::Uuid::nil());
        assert_eq!(
            select_project(&projects, Some("demo")).unwrap(),
            uuid::Uuid::nil()
        );
        assert_eq!(
            select_project(&[], Some(&uuid::Uuid::nil().to_string())).unwrap(),
            uuid::Uuid::nil()
        );
        assert!(select_project(&projects, Some("missing")).is_err());
        assert!(select_project(&[], None).is_err());
    }

    async fn mock_upload(
        state: &str,
        run_id: Option<uuid::Uuid>,
    ) -> (miette::Result<()>, Vec<Vec<u8>>) {
        use tokio::io::AsyncWriteExt;

        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let project = uuid::Uuid::nil();
        let status = serde_json::json!({"ticket": project, "project_id": project, "state": state, "run_id": run_id});
        let mut replies = vec![
            (
                200,
                serde_json::json!({"projects": [{"id": project, "name": "demo"}]}),
            ),
            (
                202,
                serde_json::json!({"ticket": project, "project_id": project, "state": "uploading"}),
            ),
        ];
        replies.extend([
                 (200, serde_json::json!({"completed":false, "part_bytes":64*1024*1024, "part_count":1,"completed_parts":[],"parts":[{"number":1,"offset":0,"size_bytes":0,"url":format!("{url}/storage/0")}]})),
                (200, serde_json::Value::Null),
                (204, serde_json::Value::Null),
                (204, serde_json::Value::Null),
            ]);
        replies.push((
            202,
            serde_json::json!({"ticket":project,"project_id":project,"state":"queued"}),
        ));
        replies.push((200, status));
        let server = tokio::spawn(async move {
            let mut requests: Vec<Vec<u8>> = Vec::new();
            for (status, mut body) in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                loop {
                    let mut buf = [0u8; 8192];
                    let n = socket.read(&mut buf).await.unwrap();
                    assert!(n > 0);
                    request.extend_from_slice(&buf[..n]);
                    if let Some(start) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&request[..start]);
                        let length = headers
                            .lines()
                            .find_map(|line| {
                                let (key, value) = line.split_once(':')?;
                                key.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if request.len() >= start + 4 + length {
                            break;
                        }
                    }
                }
                if body.get("parts").is_some() {
                    let declaration = &request;
                    let start = declaration
                        .windows(4)
                        .position(|bytes| bytes == b"\r\n\r\n")
                        .unwrap()
                        + 4;
                    let declaration: serde_json::Value =
                        serde_json::from_slice(&declaration[start..]).unwrap();
                    body["parts"][0]["size_bytes"] = declaration["size_bytes"].clone();
                }
                let body = if status == 204 {
                    String::new()
                } else {
                    body.to_string()
                };
                socket.write_all(format!("HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nETag: \"test-etag\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
                requests.push(request);
            }
            requests
        });
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("db");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("db_state"), b"abc").unwrap();
        let mut data = Vec::from(131_079u64.to_le_bytes());
        data.extend_from_slice(&[0; 16]);
        data.resize(131_079, b'x');
        std::fs::write(root.join("data"), data).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(root.join("data"))
            .unwrap()
            .set_len(8 * 1024 * 1024 * 1024)
            .unwrap();
        let ctx = AuthCtx {
            client: reqwest::Client::new(),
            issuer: url.clone(),
            api_url: url.clone(),
            creds_path: dir.path().join("credentials.json"),
        };
        tokens::Credentials {
            access_token: "test-token".into(),
            refresh_token: None,
            expires_at: tokens::now() + 1000,
            issuer: url.clone(),
            api_url: url,
        }
        .save(&ctx.creds_path)
        .unwrap();
        let result = run(
            &ctx,
            &UploadArgs {
                db_path: root.clone(),
                project: None,
                json: true,
                restart: false,
                timeout: 1,
                archive_cache: None,
                archive_cache_salt: String::new(),
                prepare_only: false,
            },
        )
        .await;
        assert_eq!(
            std::fs::metadata(root.join("data")).unwrap().len(),
            8 * 1024 * 1024 * 1024
        );
        (result, server.await.unwrap())
    }

    #[tokio::test]
    async fn uploads_only_compacted_bytes_to_storage_without_bearer_token() {
        let (result, requests) = mock_upload("succeeded", Some(uuid::Uuid::nil())).await;
        result.unwrap();
        let upload = String::from_utf8_lossy(&requests[1]);
        assert!(upload.starts_with("POST /v1/projects/00000000-0000-0000-0000-000000000000/runs "));
        assert!(upload.contains("authorization: Bearer test-token"));
        assert!(upload.contains("idempotency-key:"));
        assert!(upload.contains("content-type: application/json"));
        assert!(upload.contains("\"size_bytes\":131079"));
        assert!(!upload.contains(&"x".repeat(100)));
        assert!(requests[1].len() < 2000, "bulk data reached the API");
        assert!(
            upload.contains("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        let storage = String::from_utf8_lossy(&requests[3]);
        assert!(storage.starts_with("PUT /storage/0 "));
        assert!(!storage.contains("test-token"));
        let start = requests[3]
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .unwrap()
            + 4;
        let decoded = zstd::stream::decode_all(&requests[3][start..]).unwrap();
        assert!(
            decoded
                .windows(131_079 - 24)
                .any(|bytes| bytes.iter().all(|b| *b == b'x'))
        );
        assert!(
            requests[3].len() < 2000,
            "sparse preallocation was uploaded"
        );
        assert!(String::from_utf8_lossy(requests.last().unwrap()).contains("/run-uploads/"));
        let receipt = String::from_utf8_lossy(&requests[4]);
        assert!(receipt.contains("/archive/parts HTTP"));
        assert!(receipt.contains("\"parts\":[{\"etag\":\"\\\"test-etag\\\"\",\"number\":1}]"));
    }

    #[tokio::test]
    async fn batches_receipts_resumes_and_refreshes_failed_links() {
        use tokio::io::AsyncWriteExt;
        for retry_link in [false, true] {
            let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
                .await
                .unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let server_base = base.clone();
            let server = tokio::spawn(async move {
                let mut completed = vec![1_u32];
                let mut batches = Vec::new();
                let mut sessions = 0;
                let mut failed = false;
                loop {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let mut request = Vec::new();
                    let start = loop {
                        let mut buffer = [0; 8192];
                        let n = socket.read(&mut buffer).await.unwrap();
                        assert!(n > 0);
                        request.extend_from_slice(&buffer[..n]);
                        if let Some(start) =
                            request.windows(4).position(|bytes| bytes == b"\r\n\r\n")
                        {
                            let headers = String::from_utf8_lossy(&request[..start]);
                            let length = headers
                                .lines()
                                .find_map(|line| {
                                    let (key, value) = line.split_once(':')?;
                                    key.eq_ignore_ascii_case("content-length")
                                        .then(|| value.trim().parse::<usize>().unwrap())
                                })
                                .unwrap_or(0);
                            if request.len() >= start + 4 + length {
                                break start + 4;
                            }
                        }
                    };
                    let headers = String::from_utf8_lossy(&request[..start]);
                    let line = headers.lines().next().unwrap();
                    let path = line.split_whitespace().nth(1).unwrap();
                    let done = path.ends_with("/complete");
                    let (status, body) = if path.ends_with("/archive") {
                        assert!(headers.contains("authorization: Bearer test-token"));
                        sessions += 1;
                        let parts: Vec<_> = (1..=10_u32).filter(|number| !completed.contains(number)).map(|number| {
                            serde_json::json!({"number":number,"offset":(number-1)*4,"size_bytes":4,"url":format!("{server_base}/storage/{number}")})
                        }).collect();
                        (200, serde_json::json!({"completed":false,"part_bytes":4,"part_count":10,"completed_parts":completed,"parts":parts}).to_string())
                    } else if let Some(number) = path.strip_prefix("/storage/") {
                        assert!(!headers.contains("test-token"));
                        let number: u32 = number.parse().unwrap();
                        assert_ne!(number, 1);
                        assert_eq!(&request[start..], &[number as u8; 4]);
                        if retry_link && number == 3 && !failed {
                            failed = true;
                            (403, String::new())
                        } else {
                            (200, String::new())
                        }
                    } else if path.ends_with("/parts") {
                        assert!(headers.contains("authorization: Bearer test-token"));
                        let body: serde_json::Value =
                            serde_json::from_slice(&request[start..]).unwrap();
                        let receipts = body["parts"].as_array().unwrap();
                        assert!(!receipts.is_empty());
                        batches.push(receipts.len());
                        for receipt in receipts {
                            assert_eq!(receipt["etag"], "\"test-etag\"");
                            completed.push(receipt["number"].as_u64().unwrap() as u32);
                        }
                        (204, String::new())
                    } else {
                        assert!(done, "unexpected request {line}");
                        assert_eq!(completed.len(), 10);
                        (204, String::new())
                    };
                    socket.write_all(format!("HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nETag: \"test-etag\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
                    if done {
                        return (batches, sessions);
                    }
                }
            });
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("archive");
            let bytes: Vec<u8> = (1..=10_u8).flat_map(|number| [number; 4]).collect();
            std::fs::write(&path, bytes).unwrap();
            let ctx = AuthCtx {
                client: reqwest::Client::new(),
                issuer: base.clone(),
                api_url: base.clone(),
                creds_path: dir.path().join("credentials.json"),
            };
            tokens::Credentials {
                access_token: "test-token".into(),
                refresh_token: None,
                expires_at: tokens::now() + 1000,
                issuer: base.clone(),
                api_url: base.clone(),
            }
            .save(&ctx.creds_path)
            .unwrap();
            tokio::time::timeout(
                Duration::from_secs(20),
                transfer(
                    &ctx,
                    &base,
                    uuid::Uuid::nil(),
                    uuid::Uuid::nil(),
                    &path,
                    &Archive {
                        size_bytes: 40,
                        sha256: "a".repeat(64),
                    },
                ),
            )
            .await
            .unwrap()
            .unwrap();
            let (batches, sessions) = server.await.unwrap();
            assert_eq!(batches.iter().sum::<usize>(), 9);
            assert!(batches.iter().all(|batch| *batch <= 8));
            assert_eq!(sessions, if retry_link { 2 } else { 1 });
        }
    }

    #[tokio::test]
    async fn fails_on_quarantine_missing_run_or_publication_timeout() {
        for (state, run_id, error) in [
            ("quarantined", None, "did not publish"),
            ("succeeded", None, "omitted its run ID"),
            ("queued", None, "timed out waiting"),
        ] {
            let (result, _) = mock_upload(state, run_id).await;
            assert!(result.unwrap_err().to_string().contains(error));
        }
    }
}
