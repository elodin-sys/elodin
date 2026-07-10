//! `elodin.db` — first-class Elodin-DB client for plain Python processes.
//!
//! Exposes (under the `elodin.db` native submodule; ergonomic wrappers live in
//! `python/elodin/db.py`):
//!   * `Server` — embedded `elodin-db` instance (same server the CLI runs)
//!   * `Client` — TCP Impeller2 client: discovery, latest-value subscription,
//!     historical time-series reads, SQL (Arrow IPC) queries
//!   * `TableWriter` — batched telemetry writer: one vtable with a shared
//!     timestamp, one `Table` packet per row, bounded-queue non-blocking mode
//!
//! Threading model (as proven in `world_builder.rs`): every long-lived
//! connection lives on its own OS thread running a stellarator executor;
//! Python-facing calls communicate over bounded std channels and release the
//! GIL while blocking.

use convert_case::Casing;
use impeller2::types::PrimType;
use pyo3::prelude::*;

mod client;
mod server;
mod stream;
mod writer;

pub use client::{Client, ComponentInfo};
pub use server::Server;
pub use stream::{MsgStreamSub, StreamSub};
pub use writer::TableWriter;

/// Run a future to completion on a fresh OS thread with its own stellarator
/// executor.
///
/// `stellarator::run` consumes the calling thread's executor when it returns,
/// so it may only be used once per thread — never directly on a Python thread
/// (the second call would panic with "missing executor"). One-shot DB
/// operations (discovery, time-series reads, SQL) go through here instead;
/// long-lived connections use dedicated `stellar` threads.
pub(crate) fn block_on<T, F, Fut>(f: F) -> T
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = T> + 'static,
    T: Send + 'static,
{
    std::thread::spawn(move || stellarator::run(f))
        .join()
        .expect("stellarator worker thread panicked")
}

pub(crate) fn parse_prim_type(s: &str) -> PyResult<PrimType> {
    Ok(match s {
        "u8" => PrimType::U8,
        "u16" => PrimType::U16,
        "u32" => PrimType::U32,
        "u64" => PrimType::U64,
        "i8" => PrimType::I8,
        "i16" => PrimType::I16,
        "i32" => PrimType::I32,
        "i64" => PrimType::I64,
        "bool" => PrimType::Bool,
        "f32" => PrimType::F32,
        "f64" => PrimType::F64,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown prim type: {s:?}"
            )));
        }
    })
}

pub(crate) fn format_prim_type(prim_type: PrimType) -> String {
    match prim_type {
        PrimType::U8 => "u8",
        PrimType::U16 => "u16",
        PrimType::U32 => "u32",
        PrimType::U64 => "u64",
        PrimType::I8 => "i8",
        PrimType::I16 => "i16",
        PrimType::I32 => "i32",
        PrimType::I64 => "i64",
        PrimType::Bool => "bool",
        PrimType::F32 => "f32",
        PrimType::F64 => "f64",
    }
    .to_string()
}

/// The DataFusion table name elodin-db derives from a component name — the
/// exact conversion the server uses, so it can never drift.
#[pyfunction]
fn sql_table_name(component_name: &str) -> String {
    elodin_db::sanitize_sql_table_name(&component_name.to_case(convert_case::Case::Snake))
}

pub fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Opt-in diagnostics: ELODIN_DB_LOG=debug surfaces the embedded server's
    // and client's `tracing` events (the module otherwise stays quiet).
    if std::env::var("ELODIN_DB_LOG").is_ok() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_env("ELODIN_DB_LOG"))
            .try_init();
    }
    let child = PyModule::new(parent_module.py(), "db")?;
    child.add_class::<Server>()?;
    child.add_class::<Client>()?;
    child.add_class::<TableWriter>()?;
    child.add_class::<ComponentInfo>()?;
    child.add_class::<StreamSub>()?;
    child.add_class::<MsgStreamSub>()?;
    child.add_function(wrap_pyfunction!(sql_table_name, &child)?)?;
    parent_module.add_submodule(&child)
}
