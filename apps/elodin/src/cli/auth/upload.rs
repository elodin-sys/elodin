//! Read-only compacted streaming upload of a stopped TelemetryDB, followed by publication polling.

use std::collections::HashSet;
use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use miette::{Context, IntoDiagnostic};
use reqwest::multipart::{Form, Part};
use ring::digest::{Context as Digest, SHA256};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

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

pub async fn run(ctx: &AuthCtx, args: &UploadArgs) -> miette::Result<()> {
    tokens::ensure_valid(ctx).await?;
    let root = args.db_path.clone();
    let manifest = tokio::task::spawn_blocking(move || manifest(&root))
        .await
        .into_diagnostic()??;
    let mut creds = tokens::ensure_valid(ctx).await?;
    let base = creds.api_url.trim_end_matches('/');
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
    let mut form = Form::new().part(
        "manifest",
        Part::text(serde_json::to_string(&manifest).into_diagnostic()?)
            .mime_str("application/json")
            .into_diagnostic()?,
    );
    for entry in &manifest.files {
        let path = args.db_path.join(&entry.path);
        // Open only the current part when polled, not thousands of files at once.
        let size_bytes = entry.size_bytes;
        let stream =
            futures_util::stream::try_unfold((path, None), move |(path, file)| async move {
                let mut file = match file {
                    Some(file) => file,
                    None => tokio::fs::File::open(&path).await?.take(size_bytes),
                };
                let mut chunk = vec![0u8; 64 * 1024];
                let n = file.read(&mut chunk).await?;
                if n == 0 {
                    Ok::<_, std::io::Error>(None)
                } else {
                    chunk.truncate(n);
                    Ok(Some((chunk, (path, Some(file)))))
                }
            });
        form = form.part(
            entry.part.clone(),
            Part::stream_with_length(reqwest::Body::wrap_stream(stream), entry.size_bytes)
                .file_name(entry.part.clone())
                .mime_str("application/octet-stream")
                .into_diagnostic()?,
        );
    }
    eprintln!(
        "Uploading {} files to project {project}...",
        manifest.files.len()
    );
    let resp = ctx
        .client
        .post(&url)
        .bearer_auth(&creds.access_token)
        .header("Idempotency-Key", random_token())
        .multipart(form)
        .send()
        .await
        .into_diagnostic()
        .wrap_err("upload failed; keep the database stopped while uploading")?;
    let status_code = resp.status();
    let accepted: Status = response(resp).await?;
    if status_code != reqwest::StatusCode::ACCEPTED || accepted.project_id != project {
        return Err(miette::miette!("unexpected upload acceptance response"));
    }
    let ticket = accepted.ticket;
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
                        "upload {ticket} did not publish: {} ({})",
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
        let replies = [
            (
                200,
                serde_json::json!({"projects": [{"id": project, "name": "demo"}]}),
            ),
            (
                202,
                serde_json::json!({"ticket": project, "project_id": project, "state": "queued"}),
            ),
            (200, status),
        ];
        let server = tokio::spawn(async move {
            let mut requests = Vec::new();
            for (status, body) in replies {
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
                let body = body.to_string();
                socket.write_all(format!("HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
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
    async fn streams_manifest_first_and_waits_for_publication() {
        let (result, requests) = mock_upload("succeeded", Some(uuid::Uuid::nil())).await;
        result.unwrap();
        let upload = String::from_utf8_lossy(&requests[1]);
        assert!(upload.starts_with("POST /v1/projects/00000000-0000-0000-0000-000000000000/runs "));
        assert!(upload.contains("authorization: Bearer test-token"));
        assert!(upload.contains("idempotency-key:"));
        assert!(
            upload.find("name=\"manifest\"").unwrap() < upload.find("name=\"file_0\"").unwrap()
        );
        assert!(upload.contains("\"size_bytes\":131079"));
        assert!(upload.contains(&"x".repeat(131_079 - 24)));
        assert!(
            requests[1].len() < 140_000,
            "sparse preallocation was uploaded"
        );
        assert!(
            upload.contains("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        assert!(String::from_utf8_lossy(&requests[2]).contains("/run-uploads/"));
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
