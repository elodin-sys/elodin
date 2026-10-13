//! Elodin Cloud: project catalogs, uploads, and hosted executions over the API.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use futures::future::BoxFuture;
use serde_json::{Value, json};

use crate::exec::{HOSTED_TIMEOUT_S, Request, out_dir};
use crate::index::{Index, Run};
use crate::search;

/// A fresh bearer token (the CLI refreshes its session as needed).
pub type TokenFn = Arc<dyn Fn() -> BoxFuture<'static, anyhow::Result<String>> + Send + Sync>;

#[derive(Clone)]
pub struct Cloud {
    http: reqwest::Client,
    api: String,
    token: TokenFn,
}

#[derive(Clone, Debug)]
pub struct Project {
    pub id: String,
    pub name: String,
}

impl Cloud {
    pub fn new(api: &str, token: TokenFn) -> Self {
        let http = reqwest::Client::builder()
            .user_agent(concat!("elodin-cli/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("http client");
        Self {
            http,
            api: api.trim_end_matches('/').to_string(),
            token,
        }
    }

    async fn send(&self, req: reqwest::RequestBuilder) -> anyhow::Result<reqwest::Response> {
        let token = (self.token)().await?;
        let resp = req
            .bearer_auth(token)
            .send()
            .await
            .context("request to the Elodin Cloud API failed")?;
        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            bail!("the API rejected the session (401); run `elodin login` again");
        }
        Ok(resp)
    }

    async fn json(&self, req: reqwest::RequestBuilder) -> anyhow::Result<Value> {
        let resp = self.send(req).await?;
        let status = resp.status();
        let body: Value = resp.json().await.unwrap_or_default();
        if !status.is_success() {
            let detail = body["error"].as_str().unwrap_or("request failed");
            bail!("API error ({status}): {detail}");
        }
        Ok(body)
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.api)
    }

    pub async fn project(&self, name: &str) -> anyhow::Result<Project> {
        let body = self.json(self.http.get(self.url("/v1/projects"))).await?;
        let projects = body["projects"].as_array().cloned().unwrap_or_default();
        let found = projects
            .iter()
            .find(|p| p["name"].as_str() == Some(name))
            .or_else(|| projects.iter().find(|p| p["id"].as_str() == Some(name)));
        match found {
            Some(p) => Ok(Project {
                id: p["id"].as_str().unwrap_or_default().to_string(),
                name: p["name"].as_str().unwrap_or_default().to_string(),
            }),
            None => {
                let names: Vec<_> = projects.iter().filter_map(|p| p["name"].as_str()).collect();
                bail!(
                    "no project named {name:?}; your projects: {names:?} (create one with `elodin projects create`)"
                )
            }
        }
    }

    /// The project's complete runs, or `None` when `etag` is still current.
    pub async fn catalog(
        &self,
        project: &Project,
        etag: Option<&str>,
    ) -> anyhow::Result<Option<(Vec<Run>, String)>> {
        let mut req = self
            .http
            .get(self.url(&format!("/v1/projects/{}/catalog", project.id)));
        if let Some(e) = etag {
            req = req.header(reqwest::header::IF_NONE_MATCH, e);
        }
        let resp = self.send(req).await?;
        if resp.status() == reqwest::StatusCode::NOT_MODIFIED {
            return Ok(None);
        }
        let etag = resp
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let status = resp.status();
        let body: Value = resp.json().await.unwrap_or_default();
        if !status.is_success() {
            bail!(
                "API error ({status}): {}",
                body["error"].as_str().unwrap_or("request failed")
            );
        }
        Ok(Some((runs_of(&body)?, etag)))
    }

    pub async fn create_run(
        &self,
        project: &Project,
        body: Value,
    ) -> anyhow::Result<(reqwest::StatusCode, Value)> {
        let url = self.url(&format!("/v1/projects/{}/runs", project.id));
        let resp = self.send(self.http.post(url).json(&body)).await?;
        let status = resp.status();
        Ok((status, resp.json().await.unwrap_or_default()))
    }

    pub async fn put_part(&self, run: &str, n: usize, data: bytes::Bytes) -> anyhow::Result<Value> {
        let url = self.url(&format!("/v1/runs/{run}/parts/{n}"));
        self.json(self.http.put(url).body(data)).await
    }

    pub async fn complete(
        &self,
        run: &str,
        parts: usize,
        bytes: u64,
        sha256: &str,
    ) -> anyhow::Result<Value> {
        let url = self.url(&format!("/v1/runs/{run}/complete"));
        self.json(
            self.http
                .post(url)
                .json(&json!({"parts": parts, "bytes": bytes, "sha256": sha256})),
        )
        .await
    }

    pub async fn run(&self, run: &str) -> anyhow::Result<Value> {
        self.json(self.http.get(self.url(&format!("/v1/runs/{run}"))))
            .await
    }

    /// Start a hosted execution, wait for it, and download its artifacts into the execution
    /// directory; returns the envelope with local artifact paths.
    pub async fn execute(&self, project: &Project, req: &Request) -> anyhow::Result<Value> {
        req.validate()?;
        let url = self.url(&format!("/v1/projects/{}/executions", project.id));
        let body = json!({"code": req.code, "parameters": req.parameters, "limits": req.limits});
        let started = self.json(self.http.post(url).json(&body)).await?;
        let id = started["execution_id"]
            .as_str()
            .context("the API returned no execution_id")?
            .to_string();
        let timeout = req.limits.timeout_s.unwrap_or(HOSTED_TIMEOUT_S);
        // The API ends overdue executions itself; this only guards against a stuck API.
        let give_up = Instant::now() + Duration::from_secs(timeout + 180);
        let status = loop {
            let url = self.url(&format!("/v1/executions/{id}?wait=25"));
            let s = self.json(self.http.get(url)).await?;
            if s["state"] != "running" {
                break s;
            }
            if Instant::now() > give_up {
                bail!("execution {id} is still running after {timeout} s");
            }
        };
        let mut envelope = status["envelope"].clone();
        let dir = out_dir(req, &id)?;
        std::fs::write(dir.join("code.py"), &req.code)?;
        std::fs::write(
            dir.join("parameters.json"),
            serde_json::to_string_pretty(&req.parameters)?,
        )?;
        for a in envelope["artifacts"].as_array_mut().into_iter().flatten() {
            let name = a["name"].as_str().unwrap_or_default().to_string();
            if name.is_empty() || name.contains('/') {
                continue;
            }
            let url = self.url(&format!("/v1/executions/{id}/artifacts/{name}"));
            let resp = self.send(self.http.get(url)).await?.error_for_status()?;
            let path = dir.join(&name);
            std::fs::write(&path, resp.bytes().await?)?;
            a["path"] = json!(path);
        }
        std::fs::write(
            dir.join("envelope.json"),
            serde_json::to_string_pretty(&envelope)?,
        )?;
        Ok(envelope)
    }
}

fn runs_of(catalog: &Value) -> anyhow::Result<Vec<Run>> {
    let mut runs = Vec::new();
    for r in catalog["runs"].as_array().into_iter().flatten() {
        runs.push(Run {
            name: r["name"].as_str().unwrap_or_default().to_string(),
            id: r["id"].as_str().map(String::from),
            source_id: r["source_id"].as_str().unwrap_or_default().to_string(),
            tags: serde_json::from_value(r["tags"].clone()).unwrap_or_default(),
            catalog: serde_json::from_value(r["catalog"].clone()).context("run catalog")?,
        });
    }
    Ok(runs)
}

/// An Elodin Cloud project for the MCP server: catalog revalidated by ETag, hosted execute.
pub struct CloudBackend {
    cloud: Cloud,
    project: Project,
    root: Option<PathBuf>,
    cached: Mutex<Option<(Arc<Index>, String)>>,
}

impl CloudBackend {
    pub fn new(cloud: Cloud, project: Project, root: Option<PathBuf>) -> Self {
        Self {
            cloud,
            project,
            root,
            cached: Mutex::new(None),
        }
    }

    pub async fn index(&self) -> anyhow::Result<Arc<Index>> {
        let etag = self
            .cached
            .lock()
            .expect("cache")
            .as_ref()
            .map(|c| c.1.clone());
        match self.cloud.catalog(&self.project, etag.as_deref()).await? {
            None => Ok(self
                .cached
                .lock()
                .expect("cache")
                .as_ref()
                .expect("cached")
                .0
                .clone()),
            Some((runs, etag)) => {
                let index = Arc::new(Index::new(runs));
                *self.cached.lock().expect("cache") = Some((index.clone(), etag));
                Ok(index)
            }
        }
    }
}

impl crate::server::Backend for CloudBackend {
    fn search(&self, code: Option<&str>, limit: Option<usize>) -> anyhow::Result<Value> {
        let index = tokio::runtime::Handle::current().block_on(self.index())?;
        Ok(search::search(&index, code, limit))
    }

    fn execute(&self, mut req: Request) -> anyhow::Result<Value> {
        req.root = req.root.or_else(|| self.root.clone());
        tokio::runtime::Handle::current().block_on(self.cloud.execute(&self.project, &req))
    }
}
