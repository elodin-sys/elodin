//! Where recordings come from: local database directories or an Elodin Cloud project.

use std::path::PathBuf;

use anyhow::{Context, bail};
use sha2::{Digest, Sha256};

use crate::index::Run;

/// A local recording given as `--db [NAME=]PATH`.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalDb {
    pub name: String,
    pub path: PathBuf,
}

impl LocalDb {
    /// Parse `[NAME=]PATH`; the name defaults to the directory's basename.
    pub fn parse(arg: &str) -> anyhow::Result<Self> {
        let (name, path) = match arg.split_once('=') {
            Some((n, p)) if !n.is_empty() && !n.contains('/') => (Some(n.to_string()), p),
            _ => (None, arg),
        };
        let path = PathBuf::from(path);
        if !path.join("db_state").exists() {
            bail!("{} is not an Elodin database (no db_state)", path.display());
        }
        let path = path.canonicalize()?;
        let name = match name {
            Some(n) => n,
            None => path
                .file_name()
                .context("database path has no name")?
                .to_string_lossy()
                .into_owned(),
        };
        Ok(Self { name, path })
    }

    /// Open the recording, build its catalog, and close it again: each open component maps
    /// 16 GiB of address space, so recordings are never held open for a session.
    pub fn run(&self) -> anyhow::Result<Run> {
        let db = elodin_db::DB::open(self.path.clone())
            .with_context(|| format!("open {}", self.path.display()))?;
        let catalog = elodin_db::catalog::catalog(&db, true, None)?;
        drop(db);
        // Rows and time ranges change whenever data is appended, so they identify the content.
        let source_id = hex::encode(&Sha256::digest(catalog.to_json().as_bytes())[..16]);
        Ok(Run {
            name: self.name.clone(),
            id: None,
            source_id,
            tags: Default::default(),
            catalog,
        })
    }
}

/// Validate local databases: at least one, unique names.
pub fn local_dbs(args: &[String]) -> anyhow::Result<Vec<LocalDb>> {
    let dbs = args
        .iter()
        .map(|a| LocalDb::parse(a))
        .collect::<anyhow::Result<Vec<_>>>()?;
    for (i, a) in dbs.iter().enumerate() {
        if dbs[..i]
            .iter()
            .any(|b| b.name.eq_ignore_ascii_case(&a.name))
        {
            bail!(
                "two databases are named {:?}; rename one with --db NAME=PATH",
                a.name
            );
        }
    }
    Ok(dbs)
}

/// Local databases for the MCP server; catalogs are built once per session.
pub struct LocalBackend {
    dbs: Vec<LocalDb>,
    root: Option<PathBuf>,
    index: std::sync::Mutex<Option<std::sync::Arc<crate::index::Index>>>,
}

impl LocalBackend {
    /// `root` holds execution directories (default `./elodin-artifacts`).
    pub fn new(dbs: Vec<LocalDb>, root: Option<PathBuf>) -> Self {
        Self {
            dbs,
            root,
            index: Default::default(),
        }
    }

    pub fn index(&self) -> anyhow::Result<std::sync::Arc<crate::index::Index>> {
        let mut index = self.index.lock().expect("index lock");
        if index.is_none() {
            let runs = self
                .dbs
                .iter()
                .map(LocalDb::run)
                .collect::<anyhow::Result<_>>()?;
            *index = Some(std::sync::Arc::new(crate::index::Index::new(runs)));
        }
        Ok(index.clone().expect("index built"))
    }
}

impl crate::server::Backend for LocalBackend {
    fn search(
        &self,
        code: Option<&str>,
        limit: Option<usize>,
    ) -> anyhow::Result<serde_json::Value> {
        Ok(crate::search::search(&*self.index()?, code, limit))
    }

    fn execute(&self, mut req: crate::exec::Request) -> anyhow::Result<serde_json::Value> {
        req.root = req.root.or_else(|| self.root.clone());
        crate::exec::execute_local(&self.dbs, &self.index()?.runs, &req)
    }
}
