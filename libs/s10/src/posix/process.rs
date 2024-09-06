use futures::future::maybe_done;
use futures::{pin_mut, FutureExt};
use nu_ansi_term::Color;
use std::io::{self, stdout, Write};
use std::{collections::HashMap, path::PathBuf, process::Stdio};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::Command;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct Process {
    pub cmd: String,
    #[cfg_attr(feature = "serde", serde(default))]
    pub args: Vec<String>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub env: HashMap<String, String>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub cwd: Option<PathBuf>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub restart_policy: RestartPolicy,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum RestartPolicy {
    Never,
    #[default]
    Instant,
}

impl Process {
    pub async fn run(self, cancel_token: tokio_util::sync::CancellationToken) -> io::Result<()> {
        loop {
            let mut child = Command::new(&self.cmd);
            child
                .args(self.args.iter())
                .envs(self.env.iter())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            if let Some(cwd) = &self.cwd {
                child.current_dir(cwd);
            }
            let mut child = child.spawn()?;
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "missing child stdout"))?;
            let stderr = child
                .stderr
                .take()
                .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "missing child stderr"))?;

            let stdout_cmd = self.cmd.clone();
            let stderr_cmd = self.cmd.clone();
            tokio::spawn(
                async move { print_logs(stdout, &stdout_cmd, "stdout", Color::Blue).await },
            );
            tokio::spawn(
                async move { print_logs(stderr, &stderr_cmd, "stderr", Color::Red).await },
            );
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    child.kill().await?;
                    break;
                }
                res = child.wait() => {
                    let status = res?;
                    if let Some(code) = status.code() {
                        let color = if code == 0 {
                            Color::Green
                        }else{
                            Color::Red
                        };
                        println!("{}{}{} killed with code {}", color.paint("["), color.paint(&self.cmd), color.paint("]"), color.paint(code.to_string()))
                    }else{
                        println!("[{}] killed by signal", &self.cmd)
                    }
                    match self.restart_policy {
                        RestartPolicy::Never => {
                            return Ok(())
                        }
                        RestartPolicy::Instant => {
                            continue;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

async fn print_logs(
    input: impl AsyncRead + Unpin,
    proc_name: &str,
    log_ty: &str,
    color: nu_ansi_term::Color,
) -> io::Result<()> {
    let mut buf_reader = BufReader::new(input);
    let mut line = String::new();
    loop {
        let read_fut = maybe_done(buf_reader.read_line(&mut line));
        pin_mut!(read_fut);
        if read_fut.as_mut().now_or_never().is_none() {
            stdout().flush()?;
            read_fut.as_mut().await
        }
        let Some(res) = read_fut.take_output() else {
            return Err(io::Error::other("read_fut did not a return an output"));
        };
        if res? == 0 {
            break;
        }
        writeln!(
            stdout(),
            "{}{}{}{}{}{}",
            color.paint("["),
            color.paint(proc_name),
            color.paint(" - "),
            color.paint(log_ty),
            color.paint("] "),
            line.trim_end()
        )?;

        line.clear();
    }
    Ok(())
}
