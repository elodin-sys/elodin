//! Running analysis scripts. Locally the script runs as the user in a Python subprocess with
//! only a wall timeout, output caps, and process-group cleanup (no sandbox, by design).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::index::Run;
use crate::source::LocalDb;

pub const LOCAL_TIMEOUT_S: u64 = 600;
pub const HOSTED_TIMEOUT_S: u64 = 300;
pub const MAX_CODE: usize = 256 * 1024;
const MAX_CAPTURE: usize = 1 << 20;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Limits {
    /// Wall timeout in seconds (default 600 locally, 300 hosted).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_s: Option<u64>,
    /// Maximum rows one read may return (default 5,000,000).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_rows: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct Request {
    pub code: String,
    pub parameters: Map<String, Value>,
    pub limits: Limits,
    /// The execution directory; defaults to `<root>/<execution_id>/`.
    pub out: Option<PathBuf>,
    /// Parent of default execution directories; defaults to `./elodin-artifacts`.
    pub root: Option<PathBuf>,
}

impl Request {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.code.len() > MAX_CODE {
            bail!("code is {} bytes; the limit is {MAX_CODE}", self.code.len());
        }
        Ok(())
    }
}

/// `x-20261013T071503Z-3f9a`: sortable and unique enough for one user's executions.
pub fn execution_id() -> String {
    format!(
        "x-{}-{:04x}",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        fastrand::u16(..)
    )
}

pub fn out_dir(req: &Request, id: &str) -> anyhow::Result<PathBuf> {
    let dir = match (&req.out, &req.root) {
        (Some(o), _) => o.clone(),
        (None, Some(root)) => root.join(id),
        (None, None) => PathBuf::from("elodin-artifacts").join(id),
    };
    std::fs::create_dir_all(&dir)?;
    Ok(dir.canonicalize()?)
}

/// `ELODIN_PYTHON`, else `python3` on `PATH`.
pub fn python() -> PathBuf {
    std::env::var_os("ELODIN_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python3"))
}

/// Run `req` against local databases and return the envelope.
pub fn execute_local(dbs: &[LocalDb], runs: &[Run], req: &Request) -> anyhow::Result<Value> {
    req.validate()?;
    let id = execution_id();
    let out = out_dir(req, &id)?;
    let timeout = req.limits.timeout_s.unwrap_or(LOCAL_TIMEOUT_S);
    let sources: Vec<Value> = dbs
        .iter()
        .map(|d| {
            let run = runs.iter().find(|r| r.name == d.name);
            json!({
                "name": d.name,
                "path": d.path,
                "source_id": run.map(|r| &r.source_id),
                "catalog": run.map(|r| &r.catalog),
            })
        })
        .collect();
    let spec = json!({
        "execution_id": id,
        "code": req.code,
        "parameters": req.parameters,
        "limits": req.limits,
        "out_dir": out,
        "sources": sources,
    });

    let mut cmd = Command::new(python());
    cmd.args(["-m", "elodin.analysis._worker"])
        .env("ELODIN_CLI_VERSION", env!("CARGO_PKG_VERSION"))
        .env("MPLBACKEND", "Agg")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    let mut child = cmd
        .spawn()
        .with_context(|| format!("start {}", python().display()))?;
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(spec.to_string().as_bytes())?;
    let stdout = capture(child.stdout.take().expect("stdout"));
    let stderr = capture(child.stderr.take().expect("stderr"));

    let deadline = Instant::now() + Duration::from_secs(timeout);
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if Instant::now() >= deadline {
            kill_group(&mut child);
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let stdout = stdout.join().unwrap_or_default();
    let stderr = stderr.join().unwrap_or_default();

    let envelope_path = out.join("envelope.json");
    if status.is_some()
        && let Ok(text) = std::fs::read_to_string(&envelope_path)
    {
        return Ok(serde_json::from_str(&text)?);
    }
    // The worker died or timed out before writing an envelope.
    let tail = |s: &str| {
        s.chars()
            .rev()
            .take(2048)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<String>()
    };
    let (state, error) = match status {
        None => (
            "timeout",
            json!({"type": "Timeout", "message": format!("the script ran past {timeout} s and was stopped")}),
        ),
        Some(s) => (
            "error",
            json!({"type": "WorkerFailed", "message": format!("worker exited with {s}: {}", tail(&stderr))}),
        ),
    };
    for (name, text) in [("stdout.txt", &stdout), ("stderr.txt", &stderr)] {
        if !text.is_empty() {
            std::fs::write(out.join(name), text)?;
        }
    }
    let envelope = json!({
        "status": state,
        "execution_id": id,
        "result": null,
        "artifacts": [],
        "warnings": [],
        "error": error,
        "provenance": {},
        "usage": {},
    });
    std::fs::write(&envelope_path, serde_json::to_string_pretty(&envelope)?)?;
    Ok(envelope)
}

fn capture(mut pipe: impl Read + Send + 'static) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 8192];
        while let Ok(n) = pipe.read(&mut chunk) {
            if n == 0 {
                break;
            }
            if buf.len() < MAX_CAPTURE {
                buf.extend_from_slice(&chunk[..n.min(MAX_CAPTURE - buf.len())]);
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    })
}

/// Kill the worker and anything it spawned.
fn kill_group(child: &mut std::process::Child) {
    #[cfg(unix)]
    let _ = Command::new("kill")
        .args(["-KILL", &format!("-{}", child.id())])
        .status();
    let _ = child.kill();
    let _ = child.wait();
}

/// Parse `--param k=v` values: JSON when valid, else a string.
pub fn parse_params(file: Option<&Path>, pairs: &[String]) -> anyhow::Result<Map<String, Value>> {
    let mut params = match file {
        Some(f) => match serde_json::from_str(&std::fs::read_to_string(f)?)? {
            Value::Object(m) => m,
            _ => bail!("{} must hold a JSON object", f.display()),
        },
        None => Map::new(),
    };
    for pair in pairs {
        let (k, v) = pair
            .split_once('=')
            .with_context(|| format!("--param {pair:?} needs KEY=VALUE"))?;
        let v = serde_json::from_str(v).unwrap_or_else(|_| Value::String(v.to_string()));
        params.insert(k.to_string(), v);
    }
    Ok(params)
}
