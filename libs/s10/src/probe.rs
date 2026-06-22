use std::{
    io,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Duration,
};

use tokio::{
    fs,
    io::{AsyncBufReadExt, AsyncSeekExt, BufReader},
    net::TcpStream,
    time::{Instant, sleep},
};

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ReadyProbe {
    Tcp { addr: SocketAddr },
    File { path: PathBuf },
    Log { pattern: String },
    Delay { ms: u64 },
}

impl ReadyProbe {
    pub async fn wait(&self, log_path: Option<&Path>, timeout: Duration) -> io::Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            match self {
                ReadyProbe::Tcp { addr } => {
                    if TcpStream::connect(addr).await.is_ok() {
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
