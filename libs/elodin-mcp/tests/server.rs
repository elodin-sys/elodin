use std::sync::Arc;

use elodin_mcp::server::Server;
use elodin_mcp::source::{LocalBackend, LocalDb};
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use serde_json::json;

#[tokio::test]
async fn serves_search_and_execute() -> anyhow::Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("flight");
    elodin_db::fixture::write(&path)?;
    let db = LocalDb::parse(path.to_str().unwrap())?;

    let (client_io, server_io) = tokio::io::duplex(1 << 20);
    let backend = LocalBackend::new(vec![db], Some(tmp.path().join("out")));
    let server = Server::new(Arc::new(backend));
    tokio::spawn(async move {
        let running = server.serve(server_io).await.expect("server starts");
        let _ = running.waiting().await;
    });
    let client = ().serve(client_io).await?;

    // MCP-01: static tool definitions stay under 2,000 tokens (~4 bytes per token).
    let tools = client.list_all_tools().await?;
    let names: Vec<_> = tools.iter().map(|t| t.name.to_string()).collect();
    assert_eq!(names, ["execute", "search"]);
    let size = serde_json::to_string(&tools)?.len();
    assert!(size < 8_000, "tools/list is {size} bytes");

    let call = |name: &str, args: serde_json::Value| {
        let mut params = CallToolRequestParams::new(name.to_string());
        params.arguments = args.as_object().cloned();
        params
    };
    let found = client
        .call_tool(call("search", json!({"code": "signals('accel')"})))
        .await?;
    let out = found.structured_content.unwrap();
    assert_eq!(out["results"][0]["name"], "imu.accel");

    // Execute needs a Python with the elodin package; CI sets ELODIN_PYTHON.
    if std::env::var_os("ELODIN_PYTHON").is_some() {
        let ran = client
            .call_tool(call(
                "execute",
                json!({"code": "result = {'rows': el.signal('imu.temp').count()}"}),
            ))
            .await?;
        let env = ran.structured_content.unwrap();
        assert_eq!(env["status"], "completed", "{env}");
        assert_eq!(env["result"]["rows"], 6001);
    }
    client.cancel().await?;
    Ok(())
}
