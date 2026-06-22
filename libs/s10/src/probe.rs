use std::{
    io,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Duration,
};

#[cfg(unix)]
use tokio::net::UnixStream;
use tokio::{
    fs,
    io::{AsyncBufReadExt, AsyncSeekExt, BufReader},
    net::TcpStream,
    time::{Instant, sleep},
};

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ReadyProbe {
    Tcp { addr: String },
    Unix { path: PathBuf },
    File { path: PathBuf },
    Log { pattern: String },
    Delay { ms: u64 },
}

impl ReadyProbe {
    /// Returns a copy with `${VAR}` / `${VAR:-default}` placeholders in the
    /// probe's path/addr/pattern resolved via `lookup`. `Delay` is unchanged.
    pub fn expand(&self, lookup: impl Fn(&str) -> Option<String>) -> ReadyProbe {
        match self {
            ReadyProbe::Tcp { addr } => ReadyProbe::Tcp {
                addr: expand_env(addr, &lookup),
            },
            ReadyProbe::Unix { path } => ReadyProbe::Unix {
                path: PathBuf::from(expand_env(&path.to_string_lossy(), &lookup)),
            },
            ReadyProbe::File { path } => ReadyProbe::File {
                path: PathBuf::from(expand_env(&path.to_string_lossy(), &lookup)),
            },
            ReadyProbe::Log { pattern } => ReadyProbe::Log {
                pattern: expand_env(pattern, &lookup),
            },
            ReadyProbe::Delay { ms } => ReadyProbe::Delay { ms: *ms },
        }
    }

    pub async fn wait(&self, log_path: Option<&Path>, timeout: Duration) -> io::Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            match self {
                ReadyProbe::Tcp { addr } => {
                    let parsed = addr.parse::<SocketAddr>().map_err(|err| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("invalid tcp readiness addr `{addr}`: {err}"),
                        )
                    })?;
                    if TcpStream::connect(parsed).await.is_ok() {
                        return Ok(());
                    }
                }
                ReadyProbe::Unix { path } => {
                    if unix_connect(path).await.is_ok() {
                        return Ok(());
                    }
                }
                ReadyProbe::File { path } => {
                    if fs::metadata(path).await.is_ok() {
                        return Ok(());
                    }
                }
                ReadyProbe::Log { pattern } => {
                    let Some(path) = log_path else {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "log readiness probe requires a log_path",
                        ));
                    };
                    wait_for_log(path, pattern, deadline).await?;
                    return Ok(());
                }
                ReadyProbe::Delay { ms } => {
                    sleep(Duration::from_millis(*ms)).await;
                    return Ok(());
                }
            }

            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "readiness probe timed out",
                ));
            }
            sleep(Duration::from_millis(50)).await;
        }
    }
}

#[cfg(unix)]
async fn unix_connect(path: &Path) -> io::Result<UnixStream> {
    UnixStream::connect(path).await
}

#[cfg(not(unix))]
async fn unix_connect(_path: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "unix readiness probes are only supported on Unix",
    ))
}

async fn wait_for_log(path: &Path, pattern: &str, deadline: Instant) -> io::Result<()> {
    let mut offset = 0;
    loop {
        if let Ok(mut file) = fs::File::open(path).await {
            file.seek(std::io::SeekFrom::Start(offset)).await?;
            let mut reader = BufReader::new(file);
            let mut line = String::new();
            loop {
                let bytes = reader.read_line(&mut line).await?;
                if bytes == 0 {
                    break;
                }
                offset += bytes as u64;
                if line.contains(pattern) {
                    return Ok(());
                }
                line.clear();
            }
        }

        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "log readiness probe timed out",
            ));
        }
        sleep(Duration::from_millis(50)).await;
    }
}

pub fn parse_duration(raw: Option<&str>, default: Duration) -> io::Result<Duration> {
    let Some(raw) = raw else {
        return Ok(default);
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(default);
    }
    let number_len = raw
        .find(|ch: char| ch.is_ascii_alphabetic())
        .unwrap_or(raw.len());
    let (number, unit) = raw.split_at(number_len);
    let value = number.parse::<u64>().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid duration `{raw}`: {err}"),
        )
    })?;
    Ok(match unit {
        "" | "s" => Duration::from_secs(value),
        "ms" => Duration::from_millis(value),
        "m" => Duration::from_secs(value * 60),
        "h" => Duration::from_secs(value * 60 * 60),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported duration suffix `{unit}`"),
            ));
        }
    })
}

/// Expands shell-style `${NAME}` and `${NAME:-default}` placeholders in `input`
/// using `lookup`. An unset variable with no default expands to empty. `$$`
/// escapes a literal `$`. Nested placeholders are not supported.
pub fn expand_env(input: &str, lookup: impl Fn(&str) -> Option<String>) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' {
            if bytes.get(i + 1) == Some(&b'$') {
                out.push('$');
                i += 2;
                continue;
            }
            if bytes.get(i + 1) == Some(&b'{')
                && let Some(rel) = input[i + 2..].find('}')
            {
                let inner = &input[i + 2..i + 2 + rel];
                let (name, default) = match inner.split_once(":-") {
                    Some((name, default)) => (name, Some(default)),
                    None => (inner, None),
                };
                let value = lookup(name).unwrap_or_else(|| default.unwrap_or_default().to_string());
                out.push_str(&value);
                i = i + 2 + rel + 1;
                continue;
            }
        }
        let ch = input[i..]
            .chars()
            .next()
            .expect("byte index on char boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::expand_env;

    fn get(name: &str) -> Option<String> {
        match name {
            "PORT" => Some("9000".to_string()),
            "EMPTY" => Some(String::new()),
            _ => None,
        }
    }

    #[test]
    fn expands_set_and_default_and_escape() {
        assert_eq!(expand_env("--port=${PORT}", get), "--port=9000");
        assert_eq!(expand_env("${MISSING}", get), "");
        assert_eq!(expand_env("${MISSING:-31337}", get), "31337");
        assert_eq!(expand_env("${PORT:-31337}", get), "9000");
        // Set-but-empty still counts as set, so the default is not used.
        assert_eq!(expand_env("${EMPTY:-fallback}", get), "");
        assert_eq!(expand_env("a $$ b ${PORT}", get), "a $ b 9000");
        assert_eq!(expand_env("no placeholders", get), "no placeholders");
        // An unterminated placeholder is left verbatim.
        assert_eq!(expand_env("${UNTERMINATED", get), "${UNTERMINATED");
    }
}
