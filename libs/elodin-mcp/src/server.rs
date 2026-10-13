//! The stdio MCP server: two tools, `search` and `execute`, over any [`Backend`].

use std::sync::Arc;

use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{CallToolResult, ServerCapabilities, ServerConfig};
use rmcp::{ErrorData, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::exec::{Limits, Request};

/// Where `search` and `execute` run: local databases or an Elodin Cloud project.
pub trait Backend: Send + Sync + 'static {
    fn search(&self, code: Option<&str>, limit: Option<usize>) -> anyhow::Result<Value>;
    fn execute(&self, req: Request) -> anyhow::Result<Value>;
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    /// One expression: runs(), tables(query, run=), signals(query, run=), describe(name),
    /// sdk(query), or plain text. Empty for an overview.
    pub code: String,
    /// Maximum results for tables() and signals() (default 20).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExecuteParams {
    /// Python source. `el` and `ctx` are pre-bound; assign a small dict to `result`.
    pub code: String,
    /// Values for `ctx.parameters`, so the same code can rerun with new inputs.
    pub parameters: Option<Map<String, Value>>,
    pub limits: Option<Limits>,
}

#[derive(Clone)]
pub struct Server {
    backend: Arc<dyn Backend>,
    #[allow(dead_code, reason = "read by the tool_handler macro")]
    tool_router: ToolRouter<Self>,
}

fn internal(e: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(e.to_string(), None)
}

#[tool_router]
impl Server {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self {
            backend,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Discover runs, tables, and signals in the connected recordings, and the Python SDK reference. Pass one expression: runs(), tables(query, run=), signals(query, run=), describe(name), sdk(query), or plain text; empty code returns an overview. Returns metadata only, never sample values.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn search(
        &self,
        Parameters(p): Parameters<SearchParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let backend = self.backend.clone();
        let out = tokio::task::spawn_blocking(move || backend.search(Some(&p.code), p.limit))
            .await
            .map_err(internal)?
            .map_err(internal)?;
        Ok(CallToolResult::structured(out))
    }

    #[tool(
        description = "Run Python against the recordings. `el` is the elodin.analysis SDK; numpy, polars, pyarrow, and matplotlib are installed. Read with `el.signal(name, run=).window(start_us, end_us).to_polars()` (columns time_us, value) or `el.table(name, run=)`, save outputs with `el.artifacts.save_table/save_figure/save_json`, and assign a small JSON-serializable dict to `result`, the only output returned. `print()` goes to a log artifact. Use search first to find names.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    async fn execute(
        &self,
        Parameters(p): Parameters<ExecuteParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let backend = self.backend.clone();
        let req = Request {
            code: p.code,
            parameters: p.parameters.unwrap_or_default(),
            limits: p.limits.unwrap_or_default(),
            out: None,
            root: None,
        };
        let envelope = tokio::task::spawn_blocking(move || backend.execute(req))
            .await
            .map_err(internal)?
            .map_err(internal)?;
        Ok(if envelope["status"] == "completed" {
            CallToolResult::structured(envelope)
        } else {
            CallToolResult::structured_error(envelope)
        })
    }
}

#[tool_handler]
impl ServerHandler for Server {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Elodin recordings. Call search with empty code for an overview, then signals('...') \
             to find names, then execute Python that reads them with `el` and returns `result`.",
        )
    }
}

/// Serve MCP on stdin and stdout until the client disconnects.
pub async fn serve(backend: Arc<dyn Backend>) -> anyhow::Result<()> {
    let running = Server::new(backend).serve(rmcp::transport::stdio()).await?;
    running.waiting().await?;
    Ok(())
}
