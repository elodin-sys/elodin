//! Run EQL queries against a database file and print results (table, CSV, or binary to terminal).

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeListArray, Int32Array};
use arrow::compute;
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use futures_lite::StreamExt;
use impeller2::schema::Schema;
use miette::IntoDiagnostic;
use tabled::builder::Builder;

use crate::DB;

/// Output format for query results (always to stdout).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum QueryOutputFormat {
    /// Human-readable table (default).
    #[default]
    Table,
    /// CSV lines.
    Csv,
    /// Arrow IPC stream (binary; pipe to a file).
    ArrowIpc,
    /// Parquet (binary; pipe to a file).
    #[cfg(feature = "parquet")]
    Parquet,
}

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
    /// Output format (table, csv, parquet, or arrow-ipc to terminal).
    pub format: QueryOutputFormat,
    /// Flatten vector columns to separate columns (e.g. vel -> vel_x, vel_y, vel_z).
    pub flatten: bool,
}

/// Runs an EQL query against the database at `dbfile`, optionally limiting
/// to the first `head` or last `tail` rows, and prints the result to stdout.
pub async fn run(args: QueryArgs) -> miette::Result<()> {
    let QueryArgs {
        eql,
        dbfile,
        head,
        tail,
        format,
        flatten,
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

    let mut slice = combined.slice(start, len);
    if flatten {
        slice = flatten_record_batch(&slice)
            .map_err(|e| miette::miette!("flatten: {}", e))?;
    }

    match format {
        QueryOutputFormat::Table => print_record_batch_table(&slice)?,
        QueryOutputFormat::Csv => print_record_batch_csv(&slice)?,
        QueryOutputFormat::ArrowIpc => {
            eprintln!("Warning: output is binary; pipe to a file (e.g. ... > out.arrow)");
            print_record_batch_arrow_ipc(&slice)?;
        }
        #[cfg(feature = "parquet")]
        QueryOutputFormat::Parquet => {
            eprintln!("Warning: output is binary; pipe to a file (e.g. ... > out.parquet)");
            print_record_batch_parquet(&slice)?;
        }
    }

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

/// Flattens FixedSizeList columns into separate scalar columns.
/// Uses field metadata "element_names" (comma-separated) when present for column names (e.g. vel.x, vel.y, vel.z).
fn flatten_record_batch(batch: &RecordBatch) -> Result<RecordBatch, arrow::error::ArrowError> {
    use arrow::datatypes::FieldRef;

    let schema = batch.schema();
    let mut new_fields: Vec<FieldRef> = Vec::new();
    let mut new_columns: Vec<ArrayRef> = Vec::new();

    for (i, field) in schema.fields().iter().enumerate() {
        let column = batch.column(i);

        match field.data_type() {
            DataType::FixedSizeList(_inner_field, size) => {
                let list_array = column
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| {
                        arrow::error::ArrowError::InvalidArgumentError(
                            "expected FixedSizeListArray".to_string(),
                        )
                    })?;
                let values = list_array.values();
                let num_lists = list_array.len();
                let size = *size as usize;

                let element_name_parts: Vec<&str> = field
                    .metadata()
                    .get("element_names")
                    .map(|s| s.split(',').map(|s| s.trim()).collect())
                    .unwrap_or_default();

                for j in 0..size {
                    let indices: Vec<i32> =
                        (0..num_lists).map(|row| (row * size + j) as i32).collect();
                    let indices_array = Int32Array::from(indices);
                    let element_array = compute::take(values.as_ref(), &indices_array, None)?;
                    let suffix = if j < element_name_parts.len() && !element_name_parts[j].is_empty()
                    {
                        element_name_parts[j].to_string()
                    } else {
                        j.to_string()
                    };
                    let field_name = format!("{}.{}", field.name(), suffix);
                    new_fields.push(Arc::new(Field::new(
                        field_name,
                        element_array.data_type().clone(),
                        field.is_nullable(),
                    )));
                    new_columns.push(Arc::new(element_array));
                }
            }
            _ => {
                new_fields.push(Arc::clone(field));
                new_columns.push(Arc::clone(column));
            }
        }
    }

    let new_schema = Arc::new(arrow::datatypes::Schema::new(new_fields));
    RecordBatch::try_new(new_schema, new_columns)
}

/// Prints a RecordBatch as Arrow IPC stream to stdout.
fn print_record_batch_arrow_ipc(batch: &RecordBatch) -> miette::Result<()> {
    let mut out = std::io::stdout();
    let mut writer =
        arrow::ipc::writer::StreamWriter::try_new(&mut out, batch.schema_ref()).into_diagnostic()?;
    writer.write(batch).into_diagnostic()?;
    writer.finish().into_diagnostic()?;
    out.flush().into_diagnostic()?;
    Ok(())
}

#[cfg(feature = "parquet")]
/// Prints a RecordBatch as Parquet to stdout.
fn print_record_batch_parquet(batch: &RecordBatch) -> miette::Result<()> {
    let mut out = std::io::stdout();
    let mut writer = parquet::arrow::ArrowWriter::try_new(
        &mut out,
        batch.schema_ref().clone(),
        None,
    )
    .into_diagnostic()?;
    writer.write(batch).into_diagnostic()?;
    writer.close().into_diagnostic()?;
    out.flush().into_diagnostic()?;
    Ok(())
}

/// Prints a RecordBatch as CSV to stdout (same cell formatting as table).
fn print_record_batch_csv(batch: &RecordBatch) -> miette::Result<()> {
    let schema = batch.schema();
    let n_rows = batch.num_rows();
    let mut out = std::io::stdout();

    for (col_idx, field) in schema.fields().iter().enumerate() {
        if col_idx > 0 {
            write!(out, ",").into_diagnostic()?;
        }
        write!(out, "{}", csv_escape(field.name())).into_diagnostic()?;
    }
    writeln!(out).into_diagnostic()?;

    for i in 0..n_rows {
        for (col_idx, field) in schema.fields().iter().enumerate() {
            if col_idx > 0 {
                write!(out, ",").into_diagnostic()?;
            }
            let col = batch.column(col_idx);
            let s = format_cell(col, i, field.data_type())?;
            write!(out, "{}", csv_escape(&s)).into_diagnostic()?;
        }
        writeln!(out).into_diagnostic()?;
    }
    out.flush().into_diagnostic()?;
    Ok(())
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
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
