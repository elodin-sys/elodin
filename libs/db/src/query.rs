//! Run EQL queries against a database file and print results as a table.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use futures_lite::StreamExt;
use impeller2::schema::Schema;
use miette::IntoDiagnostic;
use tabled::builder::Builder;

use crate::DB;

/// Arguments for the `query` subcommand.
#[derive(Clone, Debug)]
pub struct QueryArgs {
    /// EQL query string.
    pub eql: String,
    /// Database directory path.
    pub dbfile: PathBuf,
    /// Show only the first N rows.
    pub head: Option<usize>,
    /// Show only the last N rows.
    pub tail: Option<usize>,
}

/// Runs an EQL query against the database at `dbfile`, optionally limiting
/// to the first `head` or last `tail` rows, and prints the result as a table.
pub async fn run(args: QueryArgs) -> miette::Result<()> {
    let QueryArgs {
        eql,
        dbfile,
        head,
        tail,
    } = args;

    if !dbfile.exists() {
        return Err(miette::miette!("database path does not exist: {}", dbfile.display()));
    }
    let db_state = dbfile.join("db_state");
    if !db_state.exists() {
        return Err(miette::miette!(
            "db_state not found in {} (not a valid elodin-db directory)",
            dbfile.display()
        ));
    }

    let db = DB::open(dbfile.clone()).into_diagnostic()?;
    let earliest = db.earliest_timestamp.latest();
    let last_ts = db.last_updated.latest();

    let sql = db.with_state(|state| {
        let components: Vec<Arc<eql::Component>> = state
            .components
            .iter()
            .filter_map(|(id, comp)| {
                let meta = state.component_metadata.get(id)?;
                let schema: Schema<Vec<u64>> = comp.schema.to_schema();
                Some(Arc::new(eql::Component::new(
                    meta.name.clone(),
                    *id,
                    schema,
                )))
            })
            .collect();
        let ctx = eql::Context::from_leaves(components, earliest, last_ts);
        ctx.sql(eql.trim()).map_err(|e| miette::miette!("EQL parse/compile error: {}", e))
    })?;

    let mut session = db.as_session_context().into_diagnostic()?;
    db.insert_views(&mut session).await.into_diagnostic()?;
    let df = session.sql(&sql).await.into_diagnostic()?;
    let mut stream = df.execute_stream().await.into_diagnostic()?;

    let mut batches = Vec::new();
    while let Some(batch) = stream.next().await {
        batches.push(batch.into_diagnostic()?);
    }

    if batches.is_empty() {
        println!("(no rows)");
        return Ok(());
    }

    let combined = arrow::compute::concat_batches(batches[0].schema_ref(), &batches)
        .into_diagnostic()?;
    let total_rows = combined.num_rows();

    let (start, len) = match (head, tail) {
        (Some(n), None) => (0, n.min(total_rows)),
        (None, Some(n)) => (total_rows.saturating_sub(n), n.min(total_rows)),
        (Some(_), Some(_)) => {
            return Err(miette::miette!("cannot use both --head and --tail"));
        }
        (None, None) => (0, total_rows),
    };

    let slice = combined.slice(start, len);
    print_record_batch_table(&slice)?;

    if len < total_rows {
        eprintln!("(showing {} of {} rows)", len, total_rows);
    }

    Ok(())
}

/// Prints a RecordBatch as a terminal table using column names and stringified values.
fn print_record_batch_table(batch: &RecordBatch) -> miette::Result<()> {
    let schema = batch.schema();
    let columns: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    let n_cols = columns.len();
    let n_rows = batch.num_rows();

    let mut builder = Builder::default();
    builder.push_record(columns.clone());
    for i in 0..n_rows {
        let mut row = Vec::with_capacity(n_cols);
        for (col_idx, field) in schema.fields().iter().enumerate() {
            let col = batch.column(col_idx);
            let s = format_cell(col, i, field.data_type())?;
            row.push(s);
        }
        builder.push_record(row);
    }

    let mut table = builder.build();
    table.with(tabled::settings::style::Style::rounded());
    println!("{}", table);

    Ok(())
}

fn format_cell(
    col: &arrow::array::ArrayRef,
    row: usize,
    data_type: &arrow::datatypes::DataType,
) -> miette::Result<String> {
    use arrow::array::*;
    use arrow::datatypes::DataType;

    if col.is_null(row) {
        return Ok("null".to_string());
    }

    let s = match data_type {
        DataType::Int8 => {
            let a = col.as_any().downcast_ref::<Int8Array>().unwrap();
            format!("{}", a.value(row))
        }
        DataType::Int16 => {
            let a = col.as_any().downcast_ref::<Int16Array>().unwrap();
            format!("{}", a.value(row))
        }
        DataType::Int32 => {
            let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
            format!("{}", a.value(row))
        }
        DataType::Int64 => {
            let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
            format!("{}", a.value(row))
        }
        DataType::UInt8 => {
            let a = col.as_any().downcast_ref::<UInt8Array>().unwrap();
            format!("{}", a.value(row))
        }
        DataType::UInt16 => {
            let a = col.as_any().downcast_ref::<UInt16Array>().unwrap();
            format!("{}", a.value(row))
        }
        DataType::UInt32 => {
            let a = col.as_any().downcast_ref::<UInt32Array>().unwrap();
            format!("{}", a.value(row))
        }
        DataType::UInt64 => {
            let a = col.as_any().downcast_ref::<UInt64Array>().unwrap();
            format!("{}", a.value(row))
        }
        DataType::Float32 => {
            let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
            format!("{}", a.value(row))
        }
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
            format!("{}", a.value(row))
        }
        DataType::Boolean => {
            let a = col.as_any().downcast_ref::<BooleanArray>().unwrap();
            format!("{}", a.value(row))
        }
        DataType::Utf8 => {
            let a = col.as_any().downcast_ref::<StringArray>().unwrap();
            a.value(row).to_string()
        }
        DataType::LargeUtf8 => {
            let a = col.as_any().downcast_ref::<LargeStringArray>().unwrap();
            a.value(row).to_string()
        }
        DataType::Timestamp(_, _) => {
            if let Some(a) = col.as_any().downcast_ref::<arrow::array::TimestampMicrosecondArray>()
            {
                format!("{}", a.value(row))
            } else if let Some(a) = col
                .as_any()
                .downcast_ref::<arrow::array::TimestampMillisecondArray>()
            {
                format!("{}", a.value(row))
            } else if let Some(a) = col
                .as_any()
                .downcast_ref::<arrow::array::TimestampNanosecondArray>()
            {
                format!("{}", a.value(row))
            } else {
                "<ts>".to_string()
            }
        }
        DataType::FixedSizeList(_, _) | DataType::List(_) => {
            let a = col.as_any().downcast_ref::<FixedSizeListArray>();
            if let Some(arr) = a {
                let vals = arr.value(row);
                let len = vals.len();
                let parts: Vec<String> = (0..len)
                    .map(|j| format_cell(&vals, j, vals.data_type()))
                    .collect::<Result<Vec<_>, _>>()?;
                format!("[{}]", parts.join(", "))
            } else {
                "<list>".to_string()
            }
        }
        _ => format!("<{:?}>", data_type),
    };
    Ok(s)
}
