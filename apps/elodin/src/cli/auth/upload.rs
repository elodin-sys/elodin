//! Resumable zstd-compressed tar uploads of a stopped TelemetryDB, followed by publication polling.

use std::collections::HashSet;
use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use miette::{Context, IntoDiagnostic};
use ring::digest::{Context as Digest, SHA256};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use super::{AuthCtx, UploadArgs, random_token, tokens};

#[derive(Debug, Serialize)]
struct Manifest {
    version: u32,
    kind: &'static str,
    files: Vec<UploadFile>,
}

#[derive(Debug, Serialize)]
struct UploadFile {
    part: String,
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
struct Archive {
    size_bytes: u64,
    sha256: String,
}

struct CheckedSource {
    file: std::io::Take<std::fs::File>,
    hash: Digest,
    bytes: u64,
}

impl Read for CheckedSource {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let n = self.file.read(buffer)?;
        self.hash.update(&buffer[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
}

fn archive(root: &Path, manifest: &Manifest) -> miette::Result<(tempfile::NamedTempFile, Archive)> {
    let output = tempfile::NamedTempFile::new().into_diagnostic()?;
    let mut encoder = zstd::stream::write::Encoder::new(output.reopen().into_diagnostic()?, 3)
        .into_diagnostic()?;
    encoder.include_checksum(true).into_diagnostic()?;
    let mut builder = tar::Builder::new(encoder);
    for entry in &manifest.files {
        let path = root.join(&entry.path);
        let file = std::fs::File::open(&path).into_diagnostic()?;
        if elodin_db::compact::compacted_file_len(&path).into_diagnostic()? != entry.size_bytes {
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
        if elodin_db::compact::compacted_file_len(&path).into_diagnostic()? != entry.size_bytes
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
    builder
        .into_inner()
        .into_diagnostic()?
        .finish()
        .into_diagnostic()?
        .sync_all()
        .into_diagnostic()?;
    let mut source = output.reopen().into_diagnostic()?;
    let size_bytes = source.metadata().into_diagnostic()?.len();
    let mut hash = Digest::new(&SHA256);
    let mut buffer = [0; 65536];
    loop {
        let n = source.read(&mut buffer).into_diagnostic()?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let sha256 = hash
        .finish()
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok((output, Archive { size_bytes, sha256 }))
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
}

#[derive(Deserialize)]
struct PartUrl {
    number: u32,
    offset: u64,
    size_bytes: u64,
    url: String,
}

pub async fn run(ctx: &AuthCtx, args: &UploadArgs) -> miette::Result<()> {
    tokens::ensure_valid(ctx).await?;
    let root = args.db_path.clone();
    let (manifest, archive_file, archive) = tokio::task::spawn_blocking(move || {
        let manifest = manifest(&root)?;
        let (file, archive) = archive(&root, &manifest)?;
        Ok::<_, miette::Report>((manifest, file, archive))
    })
    .await
    .into_diagnostic()??;
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
        transfer(ctx, &base, project, ticket, archive_file.path(), &archive).await?;
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
    tokio::time::timeout(Duration::from_secs(args.timeout), wait)
        .await
        .map_err(|_| {
            miette::miette!(
                "timed out waiting for upload {ticket}; it may still publish. Status: {status_url}"
            )
        })?
}

async fn transfer(
    ctx: &AuthCtx,
    base: &str,
    project: uuid::Uuid,
    ticket: uuid::Uuid,
    archive_path: &Path,
    archive: &Archive,
) -> miette::Result<()> {
    // Never attach the API bearer token or follow redirects on storage requests.
    let storage_client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(180))
        .build()
        .into_diagnostic()?;
    {
        let url = format!("{base}/v1/projects/{project}/run-uploads/{ticket}/archive");
        let creds = tokens::ensure_valid(ctx).await?;
        let session: FileSession = response(
            ctx.client
                .post(&url)
                .bearer_auth(&creds.access_token)
                .json(archive)
                .timeout(Duration::from_secs(30))
                .send()
                .await
                .into_diagnostic()?,
        )
        .await?;
        if session.completed {
            return Ok(());
        }
        if session.part_bytes == 0
            || session.part_bytes > 16 * 1024 * 1024
            || u64::from(session.part_count) != archive.size_bytes.div_ceil(session.part_bytes)
            || session
                .completed_parts
                .iter()
                .any(|number| *number == 0 || *number > session.part_count)
        {
            return Err(miette::miette!("invalid upload part layout"));
        }
        let mut file = tokio::fs::File::open(archive_path)
            .await
            .into_diagnostic()?;
        for number in 1..=session.part_count {
            if session.completed_parts.contains(&number) {
                continue;
            }
            let offset = u64::from(number - 1) * session.part_bytes;
            let size = (archive.size_bytes - offset).min(session.part_bytes);
            file.seek(std::io::SeekFrom::Start(offset))
                .await
                .into_diagnostic()?;
            let mut bytes = vec![0; size as usize];
            file.read_exact(&mut bytes)
                .await
                .into_diagnostic()
                .wrap_err("upload source changed; keep the database stopped")?;
            let part_url = format!("{url}/parts/{number}");
            let mut etag = None;
            for retry in 0..3 {
                let creds = tokens::ensure_valid(ctx).await?;
                let signed: PartUrl = response(
                    ctx.client
                        .post(&part_url)
                        .bearer_auth(&creds.access_token)
                        .timeout(Duration::from_secs(30))
                        .send()
                        .await
                        .into_diagnostic()?,
                )
                .await?;
                let storage_url = reqwest::Url::parse(&signed.url).into_diagnostic()?;
                if signed.number != number
                    || signed.offset != offset
                    || signed.size_bytes != size
                    || (storage_url.scheme() != "https"
                        && !(base.starts_with("http://") && storage_url.scheme() == "http"))
                    || !storage_url.username().is_empty()
                    || storage_url.password().is_some()
                {
                    return Err(miette::miette!("invalid storage upload link"));
                }
                match storage_client
                    .put(storage_url)
                    .body(bytes.clone())
                    .send()
                    .await
                {
                    Ok(resp) if resp.status().is_success() => {
                        etag = Some(
                            resp.headers()
                                .get("etag")
                                .and_then(|v| v.to_str().ok())
                                .ok_or_else(|| miette::miette!("storage omitted the part ETag"))?
                                .to_owned(),
                        );
                        break;
                    }
                    _ if retry < 2 => tokio::time::sleep(Duration::from_secs(1 << retry)).await,
                    _ => {
                        return Err(miette::miette!(
                            "storage part upload failed; rerun the same command to resume"
                        ));
                    }
                }
            }
            let creds = tokens::ensure_valid(ctx).await?;
            let resp = ctx.client.put(&part_url).bearer_auth(&creds.access_token)
                .json(&serde_json::json!({"etag": etag.ok_or_else(|| miette::miette!("missing part receipt"))?}))
                .timeout(Duration::from_secs(30)).send().await.into_diagnostic()?;
            expect_no_content(resp).await?;
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
    }
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

fn manifest(root: &Path) -> miette::Result<Manifest> {
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
    let mut files = Vec::with_capacity(paths.len());
    for path in paths {
        let path = path
            .to_str()
            .ok_or_else(|| miette::miette!("non-UTF-8 upload path"))?
            .replace(std::path::MAIN_SEPARATOR, "/");
        if !portable_path(&path) || !seen.insert(path.to_ascii_lowercase()) {
            return Err(miette::miette!(
                "nonportable or colliding upload path: {path}"
            ));
        }
        let disk_path = root.join(&path);
        let compacted_len = elodin_db::compact::compacted_file_len(&disk_path).into_diagnostic()?;
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
        kind: "telemetry_db",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_is_deterministic_compressed_and_detects_changed_sources() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("db_state"), b"abc").unwrap();
        std::fs::create_dir(dir.path().join("assets")).unwrap();
        std::fs::write(dir.path().join("assets/empty"), b"").unwrap();
        let manifest = manifest(dir.path()).unwrap();
        let (first, declaration) = archive(dir.path(), &manifest).unwrap();
        let (_, repeated) = archive(dir.path(), &manifest).unwrap();
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
        let manifest = manifest(dir.path()).unwrap();
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
                (200, serde_json::json!({"completed":false, "part_bytes":16*1024*1024, "part_count":1,"completed_parts":[]})),
                 (200, serde_json::json!({"number":1,"offset":0,"size_bytes":0,"url":format!("{url}/storage/0")})),
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
                if body.get("url").is_some() {
                    let declaration = &requests[2];
                    let start = declaration
                        .windows(4)
                        .position(|bytes| bytes == b"\r\n\r\n")
                        .unwrap()
                        + 4;
                    let declaration: serde_json::Value =
                        serde_json::from_slice(&declaration[start..]).unwrap();
                    body["size_bytes"] = declaration["size_bytes"].clone();
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
        let storage = String::from_utf8_lossy(&requests[4]);
        assert!(storage.starts_with("PUT /storage/0 "));
        assert!(!storage.contains("test-token"));
        let start = requests[4]
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .unwrap()
            + 4;
        let decoded = zstd::stream::decode_all(&requests[4][start..]).unwrap();
        assert!(
            decoded
                .windows(131_079 - 24)
                .any(|bytes| bytes.iter().all(|b| *b == b'x'))
        );
        assert!(
            requests[4].len() < 2000,
            "sparse preallocation was uploaded"
        );
        assert!(String::from_utf8_lossy(requests.last().unwrap()).contains("/run-uploads/"));
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
