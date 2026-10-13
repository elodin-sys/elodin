//! Search cards over the catalogs of every run in scope. Local and cloud runs share this.

use std::collections::BTreeMap;

use elodin_db::catalog::Catalog;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// One run's catalog, plus what the host knows about the run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Run {
    pub name: String,
    #[serde(default)]
    pub id: Option<String>,
    pub source_id: String,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
    pub catalog: Catalog,
}

const MAX_COLUMNS: usize = 40;

pub struct Index {
    pub runs: Vec<Run>,
}

impl Index {
    pub fn new(runs: Vec<Run>) -> Self {
        Self { runs }
    }

    fn scoped<'a>(&'a self, run: Option<&'a str>) -> impl Iterator<Item = &'a Run> {
        self.runs
            .iter()
            .filter(move |r| run.is_none_or(|n| r.name.eq_ignore_ascii_case(n)))
    }

    pub fn run_card(&self, run: &Run) -> Value {
        let c = &run.catalog;
        let populated = c.signals.iter().filter(|s| s.rows > 0).count();
        let first = c
            .signals
            .iter()
            .filter(|s| !s.timestamp_source)
            .filter_map(|s| s.first_us)
            .min();
        let last = c
            .signals
            .iter()
            .filter(|s| !s.timestamp_source)
            .filter_map(|s| s.last_us)
            .max();
        json!({
            "kind": "run",
            "name": run.name,
            "id": run.id,
            "source_id": run.source_id,
            "tags": run.tags,
            "join": c.join,
            "tables": c.tables.iter().filter(|t| t.rows > 0).count(),
            "signals": {"populated": populated, "empty": c.signals.len() - populated},
            "first_us": first,
            "last_us": last,
        })
    }

    /// Signal names (lowercase) and whether any run has samples.
    pub fn signal_names(&self, run: Option<&str>) -> BTreeMap<String, bool> {
        let mut names = BTreeMap::new();
        for r in self.scoped(run) {
            for s in &r.catalog.signals {
                *names.entry(s.name.to_lowercase()).or_insert(false) |= s.rows > 0;
            }
        }
        names
    }

    pub fn table_names(&self, run: Option<&str>) -> BTreeMap<String, bool> {
        let mut names = BTreeMap::new();
        for r in self.scoped(run) {
            for t in &r.catalog.tables {
                *names.entry(t.name.clone()).or_insert(false) |= t.rows > 0;
            }
        }
        names
    }

    pub fn signal_card(&self, name: &str, run: Option<&str>, full: bool) -> Option<Value> {
        let mut card: Option<Value> = None;
        let mut runs = serde_json::Map::new();
        for r in self.scoped(run) {
            let Some(s) = r.catalog.signal(name) else {
                continue;
            };
            if card.is_none() {
                let mut c = json!({
                    "kind": "signal",
                    "name": s.name.to_lowercase(),
                    "dtype": s.dtype,
                    "shape": s.shape,
                    "elements": s.elements,
                    "table": s.table,
                    "column": s.column,
                    "timestamp_source": s.timestamp_source,
                });
                if full {
                    c["original_name"] = json!(s.name);
                    c["metadata"] = json!(s.metadata);
                }
                card = Some(c);
            }
            runs.insert(
                r.name.clone(),
                json!({
                    "state": if s.rows > 0 { "populated" } else { "empty" },
                    "rows": s.rows,
                    "first_us": s.first_us,
                    "last_us": s.last_us,
                }),
            );
        }
        let mut card = card?;
        let example_run = runs.keys().next().cloned();
        card["sdk"] = json!(sdk_call(
            "signal",
            &card["name"],
            example_run.as_deref(),
            self.runs.len()
        ));
        card["runs"] = Value::Object(runs);
        Some(card)
    }

    pub fn table_card(&self, name: &str, run: Option<&str>, full: bool) -> Option<Value> {
        let mut columns: Vec<String> = Vec::new();
        let mut runs = serde_json::Map::new();
        for r in self.scoped(run) {
            let Some(t) = r.catalog.table(name) else {
                continue;
            };
            for c in &t.columns {
                if !columns.contains(c) {
                    columns.push(c.clone());
                }
            }
            runs.insert(r.name.clone(), json!({"rows": t.rows}));
        }
        if runs.is_empty() {
            return None;
        }
        let total = columns.len();
        if !full {
            columns.truncate(MAX_COLUMNS);
        }
        let name = name.to_lowercase();
        let example_run = runs.keys().next().cloned();
        Some(json!({
            "kind": "table",
            "name": name,
            "columns": columns,
            "more_columns": total - columns.len(),
            "runs": runs,
            "sdk": sdk_call("table", &json!(name), example_run.as_deref(), self.runs.len()),
        }))
    }
}

fn sdk_call(kind: &str, name: &Value, run: Option<&str>, runs_in_scope: usize) -> String {
    let name = name.as_str().unwrap_or_default();
    match run {
        Some(run) if runs_in_scope > 1 => format!("el.{kind}('{name}', run='{run}')"),
        _ => format!("el.{kind}('{name}')"),
    }
}
