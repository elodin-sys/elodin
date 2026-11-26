//! Cloud run inspection and display-name edits.

use clap::{Args, Subcommand};
use miette::IntoDiagnostic;
use serde::Deserialize;
use serde_json::Value;
use tokio::runtime::Runtime;

use crate::cli::Cli;

use super::{AuthCtx, auth::tokens, resolve_project, response};

#[derive(Args, Clone)]
pub struct RunsArgs {
    /// Project UUID or name; optional when exactly one project is available.
    #[arg(long, env = "ELODIN_PROJECT", global = true)]
    project: Option<String>,
    /// Print the raw JSON returned by the API.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: RunsCommand,
}

#[derive(Subcommand, Clone)]
enum RunsCommand {
    /// List published runs, including their saved display names
    List,
    /// Inspect a published run, including original metadata and uploader
    Get {
        /// Run UUID or exact display name within the selected project.
        run: String,
    },
    /// Rename a published run or clear its display name
    Rename {
        /// Run UUID or exact display name within the selected project.
        run: String,
        /// New display name (at most 256 Unicode characters, no control characters).
        #[arg(
            value_name = "NEW_NAME",
            required_unless_present = "clear",
            conflicts_with = "clear"
        )]
        name: Option<String>,
        /// Clear the saved display name.
        #[arg(long)]
        clear: bool,
    },
}

#[derive(Deserialize)]
struct NamedRun {
    run_id: uuid::Uuid,
    name: Option<String>,
}

#[derive(Deserialize)]
struct RunCollection {
    runs: Vec<NamedRun>,
}

fn select_run(runs: &[NamedRun], name: &str) -> miette::Result<uuid::Uuid> {
    let mut matches = runs.iter().filter(|run| run.name.as_deref() == Some(name));
    match (matches.next(), matches.next()) {
        (Some(run), None) => Ok(run.run_id),
        (None, _) => Err(miette::miette!(
            "run '{name}' not found in the selected project"
        )),
        _ => Err(miette::miette!("run '{name}' is ambiguous; use its UUID")),
    }
}

async fn resolve_run(
    ctx: &AuthCtx,
    creds: &tokens::Credentials,
    url: &str,
    requested: &str,
) -> miette::Result<uuid::Uuid> {
    if let Ok(id) = requested.parse() {
        return Ok(id);
    }
    let collection: RunCollection = response(
        ctx.client
            .get(url)
            .bearer_auth(&creds.access_token)
            .send()
            .await
            .into_diagnostic()?,
    )
    .await?;
    select_run(&collection.runs, requested)
}

async fn run(ctx: &AuthCtx, args: &RunsArgs) -> miette::Result<()> {
    let creds = tokens::ensure_valid(ctx).await?;
    let project = resolve_project(ctx, &creds, args.project.as_deref()).await?;
    let url = format!(
        "{}/v1/projects/{project}/runs",
        creds.api_url.trim_end_matches('/')
    );
    let request = match &args.command {
        RunsCommand::List => ctx.client.get(&url),
        RunsCommand::Get { run } => {
            let id = resolve_run(ctx, &creds, &url, run).await?;
            ctx.client.get(format!("{url}/{id}"))
        }
        RunsCommand::Rename { run, name, .. } => {
            let id = resolve_run(ctx, &creds, &url, run).await?;
            ctx.client
                .patch(format!("{url}/{id}"))
                .json(&serde_json::json!({"name": name}))
        }
    };
    let value: Value = response(
        request
            .bearer_auth(&creds.access_token)
            .send()
            .await
            .into_diagnostic()?,
    )
    .await?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&value).into_diagnostic()?
        );
    } else {
        println!("{}", display(&value, &args.command)?);
    }
    Ok(())
}

fn display(value: &Value, command: &RunsCommand) -> miette::Result<String> {
    fn summary(run: &Value) -> miette::Result<String> {
        let id = run
            .get("run_id")
            .and_then(Value::as_str)
            .ok_or_else(|| miette::miette!("invalid API response: missing run_id"))?;
        let name = run
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("(unnamed)");
        Ok(format!("{name}  ({id})"))
    }
    match command {
        RunsCommand::List => {
            let runs = value
                .get("runs")
                .and_then(Value::as_array)
                .ok_or_else(|| miette::miette!("invalid API response: missing runs"))?;
            if runs.is_empty() {
                return Ok("No runs yet.".into());
            }
            runs.iter()
                .map(summary)
                .collect::<miette::Result<Vec<_>>>()
                .map(|rows| rows.join("\n"))
        }
        RunsCommand::Get { .. } => Ok(format!(
            "{}\n{}",
            summary(value)?,
            serde_json::to_string_pretty(value).into_diagnostic()?
        )),
        RunsCommand::Rename { .. } => Ok(format!("Saved run: {}", summary(value)?)),
    }
}

impl Cli {
    pub fn runs(self, args: RunsArgs, rt: Runtime) -> miette::Result<()> {
        let ctx = self.auth_ctx()?;
        rt.block_on(run(&ctx, &args))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn rename_requires_a_name_or_clear() {
        let id = uuid::Uuid::nil().to_string();
        for tail in [vec![], vec!["New name", "--clear"]] {
            let mut args = vec!["elodin", "runs", "rename", &id];
            args.extend(tail);
            assert!(Cli::try_parse_from(args).is_err());
        }
        for tail in [vec!["New name"], vec!["--clear"], vec![""]] {
            let mut args = vec!["elodin", "runs", "rename", &id];
            args.extend(tail);
            args.extend(["--project", "demo", "--json"]);
            assert!(Cli::try_parse_from(args).is_ok());
        }
        assert!(Cli::try_parse_from(["elodin", "runs", "get", "Flight 14"]).is_ok());
        assert!(Cli::try_parse_from(["elodin", "runs", "rename", "Flight 14", "--clear"]).is_ok());
    }

    #[test]
    fn names_are_exact_case_sensitive_and_must_identify_one_run() {
        let id = uuid::Uuid::new_v4();
        let mut runs = vec![
            NamedRun {
                run_id: id,
                name: Some("Flight / 測試 ? #14".into()),
            },
            NamedRun {
                run_id: uuid::Uuid::new_v4(),
                name: None,
            },
        ];
        assert_eq!(select_run(&runs, "Flight / 測試 ? #14").unwrap(), id);
        for missing in ["flight / 測試 ? #14", "Flight", "", " Flight / 測試 ? #14 "] {
            assert!(
                select_run(&runs, missing)
                    .unwrap_err()
                    .to_string()
                    .contains("not found")
            );
        }
        runs.push(NamedRun {
            run_id: uuid::Uuid::new_v4(),
            name: Some("Flight / 測試 ? #14".into()),
        });
        assert!(
            select_run(&runs, "Flight / 測試 ? #14")
                .unwrap_err()
                .to_string()
                .contains("ambiguous")
        );
    }

    #[test]
    fn lists_saved_names_and_unnamed_ids() {
        let runs = serde_json::json!({"runs": [
            {"run_id": "first", "name": "Drop Test"},
            {"run_id": "second", "name": null}
        ]});
        assert_eq!(
            display(&runs, &RunsCommand::List).unwrap(),
            "Drop Test  (first)\n(unnamed)  (second)"
        );
        assert_eq!(
            display(&serde_json::json!({"runs": []}), &RunsCommand::List).unwrap(),
            "No runs yet."
        );
    }

    #[tokio::test]
    async fn commands_use_authenticated_project_scoped_endpoints() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let project = uuid::Uuid::new_v4();
        let run_id = uuid::Uuid::new_v4();
        for by_name in [false, true] {
            let selector = if by_name {
                "Flight / 測試 ? #14".to_owned()
            } else {
                run_id.to_string()
            };
            for (command, method, body, status) in [
                (RunsCommand::List, "GET", None, 200),
                (
                    RunsCommand::Get {
                        run: selector.clone(),
                    },
                    "GET",
                    None,
                    200,
                ),
                (
                    RunsCommand::Rename {
                        run: selector.clone(),
                        name: Some("  New name  ".into()),
                        clear: false,
                    },
                    "PATCH",
                    Some(serde_json::json!({"name": "  New name  "})),
                    200,
                ),
                (
                    RunsCommand::Rename {
                        run: selector.clone(),
                        name: None,
                        clear: true,
                    },
                    "PATCH",
                    Some(serde_json::json!({"name": null})),
                    200,
                ),
                (
                    RunsCommand::Rename {
                        run: selector.clone(),
                        name: Some("taken".into()),
                        clear: false,
                    },
                    "PATCH",
                    Some(serde_json::json!({"name": "taken"})),
                    409,
                ),
                (
                    RunsCommand::Get {
                        run: selector.clone(),
                    },
                    "GET",
                    None,
                    401,
                ),
                (
                    RunsCommand::Get {
                        run: selector.clone(),
                    },
                    "GET",
                    None,
                    404,
                ),
                (
                    RunsCommand::Rename {
                        run: selector.clone(),
                        name: Some("bad\nname".into()),
                        clear: false,
                    },
                    "PATCH",
                    Some(serde_json::json!({"name": "bad\nname"})),
                    400,
                ),
                (
                    RunsCommand::Rename {
                        run: selector.clone(),
                        name: None,
                        clear: true,
                    },
                    "PATCH",
                    Some(serde_json::json!({"name": null})),
                    403,
                ),
            ] {
                let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
                    .await
                    .unwrap();
                let url = format!("http://{}", listener.local_addr().unwrap());
                let list = matches!(command, RunsCommand::List);
                let server = tokio::spawn(async move {
                    for index in 0..if by_name && !list { 3 } else { 2 } {
                        let (mut socket, _) = listener.accept().await.unwrap();
                        let mut request = Vec::new();
                        let (start, headers) = loop {
                            let mut buf = [0; 4096];
                            let n = socket.read(&mut buf).await.unwrap();
                            assert!(n > 0);
                            request.extend_from_slice(&buf[..n]);
                            if let Some(start) = request.windows(4).position(|b| b == b"\r\n\r\n") {
                                let headers = String::from_utf8(request[..start].to_vec()).unwrap();
                                let length = headers
                                    .lines()
                                    .find_map(|line| {
                                        let (key, value) = line.split_once(':')?;
                                        key.eq_ignore_ascii_case("content-length")
                                            .then(|| value.trim().parse::<usize>().unwrap())
                                    })
                                    .unwrap_or(0);
                                if request.len() >= start + 4 + length {
                                    break (start + 4, headers);
                                }
                            }
                        };
                        assert!(headers.contains("authorization: Bearer test-token"));
                        let (code, reply) = if index == 0 {
                            assert!(headers.starts_with("GET /v1/projects "));
                            (
                                200,
                                serde_json::json!({"projects": [{"id": project, "name": "demo"}]}),
                            )
                        } else if by_name && !list && index == 1 {
                            assert!(
                                headers.starts_with(&format!("GET /v1/projects/{project}/runs "))
                            );
                            (
                                200,
                                serde_json::json!({"runs": [{"run_id": run_id, "name": "Flight / 測試 ? #14"}]}),
                            )
                        } else {
                            let suffix = if list {
                                String::new()
                            } else {
                                format!("/{run_id}")
                            };
                            assert!(headers.starts_with(&format!(
                                "{method} /v1/projects/{project}/runs{suffix} "
                            )));
                            if let Some(body) = &body {
                                assert_eq!(
                                    &serde_json::from_slice::<Value>(&request[start..]).unwrap(),
                                    body
                                );
                            }
                            (
                                status,
                                if status == 409 {
                                    serde_json::json!({"error": "run_name_conflict"})
                                } else {
                                    serde_json::json!({"run_id": run_id, "name": "New name"})
                                },
                            )
                        };
                        let reply = reply.to_string();
                        socket.write_all(format!("HTTP/1.1 {code} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len()).as_bytes()).await.unwrap();
                    }
                });
                let dir = tempfile::tempdir().unwrap();
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
                let result = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    run(
                        &ctx,
                        &RunsArgs {
                            project: Some("demo".into()),
                            json: true,
                            command,
                        },
                    ),
                )
                .await
                .unwrap();
                server.await.unwrap();
                if status == 200 {
                    result.unwrap();
                } else {
                    let error = result.unwrap_err().to_string();
                    assert!(error.contains(&status.to_string()));
                    if status == 409 {
                        assert!(error.contains("run_name_conflict"));
                    }
                    if status == 401 {
                        assert!(error.contains("elodin login"));
                    }
                }
            }
        }
    }
}
