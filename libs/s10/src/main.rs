use std::io;

use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::io::Result<()> {
    let config = Config::parse()?;
    println!("starting with config {:#?}", config);
    let cancel_token = CancellationToken::new();
    let mut tasks: JoinSet<_> = config
        .tasks
        .into_iter()
        .map(|t| {
            let token = cancel_token.child_token();
            async {
                match t {
                    s10::Task::Process(p) => p.run(token).await,
                    s10::Task::Watch(w) => w.run(token).await,
                }
            }
        })
        .collect();
    loop {
        tokio::select! {
            res = tasks.join_next() => {
                if res.is_none() {
                    println!("all tasks have ended exiting");
                    return Ok(())
                }
            }
            _ = tokio::signal::ctrl_c() => {
                println!("killing processes");
                cancel_token.cancel();
                return Ok(())
            }
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct Config {
    tasks: Vec<s10::Task>,
}

impl Config {
    pub fn parse() -> io::Result<Self> {
        let config_paths = [
            std::env::var("S10_CONFIG").unwrap_or_else(|_| "/etc/elodin/s10.toml".to_string()),
            "./config.toml".to_string(),
        ];
        for path in config_paths {
            let Ok(config) = std::fs::read_to_string(path) else {
                continue;
            };
            return toml::from_str(&config).map_err(io::Error::other);
        }
        Err(io::Error::other("no config file found"))
    }
}
