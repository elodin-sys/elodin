//! List all component names in a database.

use std::path::PathBuf;

use impeller2::types::Timestamp;
use tabular::{Row, Table};

use crate::{DB, Error};

/// Row for long-format output: name, first timestamp, last timestamp, entry count.
struct ComponentRow {
    name: String,
    first_ts: Option<Timestamp>,
    last_ts: Option<Timestamp>,
    count: usize,
}

/// Lists all component names in the database at `path`, one per line, sorted.
/// If `long` is true, also prints first timestamp, last timestamp, and entry count per component.
pub fn run(path: PathBuf, long: bool) -> Result<(), Error> {
    if !path.exists() {
        return Err(Error::MissingDbState(path.join("db_state")));
    }
    let db_state = path.join("db_state");
    if !db_state.exists() {
        return Err(Error::MissingDbState(db_state));
    }

    let db = DB::open(path)?;

    if long {
        let mut rows: Vec<ComponentRow> = db.with_state(|state| {
            state
                .components
                .iter()
                .filter_map(|(id, component)| {
                    let meta = state.component_metadata.get(id)?;
                    let first_ts = if component.time_series.index().is_empty() {
                        None
                    } else {
                        Some(component.time_series.start_timestamp())
                    };
                    let last_ts = component.time_series.latest().map(|(t, _)| *t);
                    let count = component.time_series.sample_count();
                    Some(ComponentRow {
                        name: meta.name.clone(),
                        first_ts,
                        last_ts,
                        count,
                    })
                })
                .collect::<Vec<_>>()
        });
        rows.sort_by(|a, b| a.name.cmp(&b.name));

        let mut table = Table::new("{:<}  {:<}  {:<}  {:>}").with_row(
            Row::new()
                .with_cell("component")
                .with_cell("first")
                .with_cell("last")
                .with_cell("entries"),
        );
        for row in rows {
            let first_str = row
                .first_ts
                .map(|ts| hifitime::Epoch::from(ts).to_string())
                .unwrap_or_else(|| "-".to_string());
            let last_str = row
                .last_ts
                .map(|ts| hifitime::Epoch::from(ts).to_string())
                .unwrap_or_else(|| "-".to_string());
            table.add_row(
                Row::new()
                    .with_cell(row.name)
                    .with_cell(first_str)
                    .with_cell(last_str)
                    .with_cell(row.count.to_string()),
            );
        }
        print!("{table}");
    } else {
        let mut names: Vec<String> = db.with_state(|state| {
            state
                .component_metadata
                .values()
                .map(|m| m.name.clone())
                .collect::<Vec<_>>()
        });
        names.sort();

        for name in names {
            println!("{name}");
        }
    }
    Ok(())
}
