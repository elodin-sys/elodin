//! Query component data from a database file and print results (table, CSV, or binary to terminal).

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

/// Decimal places for float display; "full" means no rounding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Show floats with this many decimal places.
    Decimals(u32),
    /// Show full precision (no rounding).
    Full,
}

impl Default for Precision {
    fn default() -> Self {
        Precision::Decimals(6)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RowDescription {
    /// Number of rows (negative => from end)
    Count(i64),
    /// Duration in seconds (e.g. "1.2s"; negative => from end, e.g. -1s = 1s before last entry)
    Second(f64),
    /// Duration in milliseconds (e.g. "3400ms"; negative => from end)
    Millisecond(i64),
    /// Duration in microseconds (e.g. "560000µs"; negative => from end)
    Microsecond(i64),
    /// Duration in nanoseconds (e.g. "78000000ns"; negative => from end)
    Nanosecond(i64),
}

impl RowDescription {
    /// Converts a duration variant to microseconds for time-column comparison. Returns None for Count.
    fn to_microseconds(&self) -> Option<f64> {
        match self {
            RowDescription::Count(_) => None,
            RowDescription::Second(s) => Some(s * 1_000_000.0),
            RowDescription::Millisecond(ms) => Some(*ms as f64 * 1_000.0),
            RowDescription::Microsecond(us) => Some(*us as f64),
            RowDescription::Nanosecond(ns) => Some(*ns as f64 / 1_000.0),
        }
    }
}

impl std::str::FromStr for RowDescription {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err("empty row description".to_string());
        }
        if let Some(rest) = s.strip_suffix("ms") {
            if rest.is_empty() {
                return Err("invalid duration: 'ms' needs a number".to_string());
            }
            let n: i64 = rest
                .parse()
                .map_err(|_| format!("invalid milliseconds: '{}'", s))?;
            return Ok(RowDescription::Millisecond(n));
        }
        if let Some(us) = s.strip_suffix("us").or_else(|| s.strip_suffix("µs")) {
            if us.is_empty() {
                return Err("invalid duration: 'us' needs a number".to_string());
            }
            let n: i64 = us
                .parse()
                .map_err(|_| format!("invalid microseconds: '{}'", s))?;
            return Ok(RowDescription::Microsecond(n));
        }
        if let Some(ns) = s.strip_suffix("ns") {
            if ns.is_empty() {
                return Err("invalid duration: 'ns' needs a number".to_string());
            }
            let n: i64 = ns
                .parse()
                .map_err(|_| format!("invalid nanoseconds: '{}'", s))?;
            return Ok(RowDescription::Nanosecond(n));
        }
        if let Some(rest) = s.strip_suffix('s') {
            if rest.is_empty() {
                return Err("invalid duration: 's' needs a number".to_string());
            }
            let n: f64 = rest
                .parse()
                .map_err(|_| format!("invalid seconds: '{}'", s))?;
            return Ok(RowDescription::Second(n));
        }
        let n: i64 = s
            .parse()
            .map_err(|_| format!(
                "row description must be an integer or duration (e.g. 10, 2.6s, 340000ms, 53000ns), got '{}'",
                s
            ))?;
        Ok(RowDescription::Count(n))
    }
}

impl std::str::FromStr for Precision {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("full") {
            return Ok(Precision::Full);
        }
        s.parse::<u32>()
            .map(Precision::Decimals)
            .map_err(|_| format!("precision must be a number or 'full', got '{}'", s))
    }
}

/// How to display the time column in table/CSV output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum TimeFormat {
    /// Do not show the time column.
    #[clap(alias = "none")]
    Omit,
    /// Show as date-time (e.g. 2025-02-27T12:00:00.000 UTC, alias 'dt').
    #[clap(alias = "dt")]
    Datetime,
    /// Show as seconds since epoch (default, alias 's').
    #[default]
    #[clap(alias = "s")]
    Seconds,
    /// Show as milliseconds since epoch (alias 'ms').
    #[clap(alias = "ms")]
    Milliseconds,
    /// Show as microseconds since epoch (alias 'µs' and 'us').
    #[clap(alias = "us", alias = "µs")]
    Microseconds,
}

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
    /// EQL query (e.g. component name). Mutually exclusive with `sql`.
    pub eql: Option<String>,
    /// Raw SQL query. Mutually exclusive with `eql`.
    pub sql: Option<String>,
    /// If true, print the SQL (EQL conversion or raw) to stderr.
    pub verbose: bool,
    /// Decimal places for floats (number or "full"); default 6.
    pub precision: Precision,
    /// Database directory path.
    pub dbfile: PathBuf,
    /// Skip rows from the start (integer or duration, e.g. 10 or 2.6s; negative int = from end).
    pub offset: Option<RowDescription>,
    /// Return at most this many rows or this duration (e.g. 10 or 2.6s).
    pub limit: Option<RowDescription>,
    /// Output format (table, csv, parquet, or arrow-ipc to terminal).
    pub format: QueryOutputFormat,
    /// Flatten vector columns to separate columns (e.g. vel -> vel_x, vel_y, vel_z).
    pub flatten: bool,
    /// How to display the time column. If None, inferred from --offset/--limit duration unit, else seconds.
    pub time_format: Option<TimeFormat>,
    /// If true, add a first column with the row index (0-based).
    pub row_index: bool,
}

/// Infers time display format from offset/limit when they are duration variants.
fn time_format_from_offset_limit(
    offset: Option<&RowDescription>,
    limit: Option<&RowDescription>,
) -> Option<TimeFormat> {
    let from = |rd: &RowDescription| match rd {
        RowDescription::Second(_) => Some(TimeFormat::Seconds),
        RowDescription::Millisecond(_) => Some(TimeFormat::Milliseconds),
        RowDescription::Microsecond(_) => Some(TimeFormat::Microseconds),
        RowDescription::Nanosecond(_) => Some(TimeFormat::Microseconds),
        RowDescription::Count(_) => None,
    };
    offset.and_then(from).or_else(|| limit.and_then(from))
}

/// Queries the database at `dbfile` and prints the result to stdout.
pub async fn run(args: QueryArgs) -> miette::Result<()> {
    let QueryArgs {
        eql,
        sql,
        verbose,
        precision,
        dbfile,
        offset,
        limit,
        format,
        flatten,
        time_format,
        row_index,
    } = args;

    let time_format = time_format
        .or_else(|| time_format_from_offset_limit(offset.as_ref(), limit.as_ref()))
        .unwrap_or(TimeFormat::Seconds);

    if !dbfile.exists() {
        return Err(miette::miette!(
            "database path does not exist: {}",
            dbfile.display()
        ));
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

    let sql = match (&eql, &sql) {
        (Some(eql), None) => db.with_state(|state| {
            let components: Vec<Arc<eql::Component>> = state
                .components
                .iter()
                .filter_map(|(id, comp)| {
                    let meta = state.component_metadata.get(id)?;
                    let schema: Schema<Vec<u64>> = comp.schema.to_schema();
                    let element_names: Vec<String> = meta
                        .element_names()
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    let component = if element_names.is_empty() {
                        eql::Component::new(meta.name.clone(), *id, schema)
                    } else {
                        eql::Component::new_with_element_names(
                            meta.name.clone(),
                            *id,
                            schema,
                            element_names,
                        )
                    };
                    Some(Arc::new(component))
                })
                .collect();
            let ctx = eql::Context::from_leaves(components, earliest, last_ts);
            let sql = if time_format == TimeFormat::Omit {
                ctx.sql(eql.trim())
            } else {
                ctx.sql_with_options(
                    eql.trim(),
                    &eql::SqlOptions {
                        include_time_column: true,
                    },
                )
            };
            sql.inspect(|sql| {
                if verbose {
                    eprintln!("EQL to SQL: {}", sql);
                }
            })
            .map_err(|e| miette::miette!("query parse error: {}", e))
        })?,
        (None, Some(s)) => s.clone(),
        (Some(_), Some(_)) => {
            return Err(miette::miette!(
                "cannot use both --eql and --sql; specify exactly one"
            ));
        }
        (None, None) => {
            return Err(miette::miette!("must specify either --eql or --sql"));
        }
    };

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

    let combined =
        arrow::compute::concat_batches(batches[0].schema_ref(), &batches).into_diagnostic()?;
    let total_rows = combined.num_rows();

    let (start, len) = resolve_slice(&combined, total_rows, offset.as_ref(), limit.as_ref())?;

    let mut slice = combined.slice(start, len);
    if flatten {
        slice = flatten_record_batch(&slice).map_err(|e| miette::miette!("flatten: {}", e))?;
    }
    slice = time_column_first(&slice).into_diagnostic()?;

    let is_table_or_csv = matches!(format, QueryOutputFormat::Table | QueryOutputFormat::Csv);
    if is_table_or_csv && let Precision::Decimals(p) = precision {
        eprintln!(
            "Note: numeric columns shown with limited precision: {} decimal places. Use '--precision full' to see all digits.",
            p,
        );
    }

    match format {
        QueryOutputFormat::Table => print_record_batch_table(
            &slice,
            time_format,
            &precision,
            row_index,
            start,
        )?,
        QueryOutputFormat::Csv => print_record_batch_csv(
            &slice,
            time_format,
            &precision,
            row_index,
            start,
        )?,
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

/// Resolves (start, len) from optional offset/limit (row count or duration).
fn resolve_slice(
    batch: &RecordBatch,
    total_rows: usize,
    offset: Option<&RowDescription>,
    limit: Option<&RowDescription>,
) -> miette::Result<(usize, usize)> {
    if offset.is_none() && limit.is_none() {
        return Ok((0, total_rows));
    }
    let time_us = time_column_microseconds(batch);
    let needs_time = offset
        .and_then(RowDescription::to_microseconds)
        .is_some()
        || limit.and_then(RowDescription::to_microseconds).is_some();
    if needs_time && time_us.is_none() {
        return Err(miette::miette!(
            "duration for --offset/--limit requires a time column; use --eql or ensure the query selects time"
        ));
    }
    let times = time_us.as_ref();
    let start = if let Some(off) = offset {
        match off {
            RowDescription::Count(n) => {
                if *n >= 0 {
                    (*n as usize).min(total_rows)
                } else {
                    let from_end = (total_rows as i64).saturating_add(*n).max(0) as usize;
                    from_end.min(total_rows)
                }
            }
            _ => {
                let offset_us = off.to_microseconds().unwrap();
                let times = times.unwrap();
                let first_us = times[0];
                let last_us = times[total_rows - 1];
                if offset_us >= 0.0 {
                    times
                        .iter()
                        .position(|&t| (t as f64) >= first_us as f64 + offset_us)
                        .unwrap_or(total_rows)
                        .min(total_rows)
                } else {
                    // Negative duration: from end (e.g. -1s = 1 second before last entry)
                    let threshold = last_us as f64 + offset_us;
                    times
                        .iter()
                        .position(|&t| (t as f64) >= threshold)
                        .unwrap_or(0)
                        .min(total_rows)
                }
            }
        }
    } else {
        0
    };
    let remaining = total_rows.saturating_sub(start);
    let (final_start, len) = if let Some(lim) = limit {
        match lim {
            RowDescription::Count(n) => (start, ((*n).max(0) as usize).min(remaining)),
            _ => {
                let limit_us = lim.to_microseconds().unwrap();
                let times = times.unwrap();
                let last_us = times[total_rows - 1];
                if limit_us >= 0.0 {
                    let start_time_us = times[start];
                    let mut end = start;
                    while end < total_rows && (times[end] as f64) < start_time_us as f64 + limit_us {
                        end += 1;
                    }
                    (start, (end - start).min(remaining))
                } else {
                    // Negative duration: last |limit_us| of data (e.g. -1s = take last 1 second)
                    let threshold = last_us as f64 + limit_us;
                    let start_from_end = times
                        .iter()
                        .position(|&t| (t as f64) >= threshold)
                        .unwrap_or(0)
                        .min(total_rows);
                    let effective_start = start.max(start_from_end);
                    let len = (total_rows - effective_start).min(total_rows);
                    (effective_start, len)
                }
            }
        }
    } else {
        (start, remaining)
    };
    Ok((final_start, len))
}

/// Returns the "time" column as microseconds per row, or None if missing/unsupported type.
fn time_column_microseconds(batch: &RecordBatch) -> Option<Vec<i64>> {
    use arrow::array::Array;
    let schema = batch.schema();
    let time_idx = schema.fields().iter().position(|f| f.name() == "time")?;
    let col = batch.column(time_idx);
    let n = col.len();
    let mut out = Vec::with_capacity(n);
    if let Some(a) = col.as_any().downcast_ref::<arrow::array::TimestampMicrosecondArray>() {
        for i in 0..n {
            out.push(a.value(i));
        }
        return Some(out);
    }
    if let Some(a) = col.as_any().downcast_ref::<arrow::array::TimestampMillisecondArray>() {
        for i in 0..n {
            out.push(a.value(i).saturating_mul(1000));
        }
        return Some(out);
    }
    if let Some(a) = col.as_any().downcast_ref::<arrow::array::TimestampNanosecondArray>() {
        for i in 0..n {
            out.push(a.value(i) / 1000);
        }
        return Some(out);
    }
    None
}

/// Returns the display name for a column; appends the unit for the time column.
fn column_header_with_unit(name: &str, time_format: TimeFormat) -> String {
    if name != "time" {
        return name.to_string();
    }
    match time_format {
        TimeFormat::Omit => name.to_string(),
        TimeFormat::Seconds => "time (s)".to_string(),
        TimeFormat::Milliseconds => "time (ms)".to_string(),
        TimeFormat::Microseconds => "time (μs)".to_string(),
        TimeFormat::Datetime => "time (UTC)".to_string(),
    }
}

/// Reorders columns so "time" is first if present.
fn time_column_first(batch: &RecordBatch) -> Result<RecordBatch, arrow::error::ArrowError> {
    let schema = batch.schema();
    let time_idx = schema.fields().iter().position(|f| f.name() == "time");
    let Some(time_idx) = time_idx else {
        return Ok(batch.clone());
    };
    if time_idx == 0 {
        return Ok(batch.clone());
    }
    let mut indices: Vec<usize> = (0..schema.fields().len()).collect();
    indices.remove(time_idx);
    indices.insert(0, time_idx);
    let new_columns: Vec<ArrayRef> = indices.iter().map(|&i| batch.column(i).clone()).collect();
    let new_fields: Vec<Arc<Field>> = indices
        .iter()
        .map(|&i| Arc::clone(&schema.fields()[i]))
        .collect();
    let new_schema = Arc::new(arrow::datatypes::Schema::new(new_fields));
    RecordBatch::try_new(new_schema, new_columns)
}

/// Prints a RecordBatch as a terminal table using column names and stringified values.
fn print_record_batch_table(
    batch: &RecordBatch,
    time_format: TimeFormat,
    precision: &Precision,
    row_index: bool,
    row_index_start: usize,
) -> miette::Result<()> {
    let schema = batch.schema();
    let col_indices: Vec<(usize, &Arc<Field>)> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| time_format != TimeFormat::Omit || f.name() != "time")
        .collect();
    let mut columns: Vec<String> = col_indices
        .iter()
        .map(|(_, f)| column_header_with_unit(f.name(), time_format))
        .collect();
    if row_index {
        columns.insert(0, "index".to_string());
    }
    let n_rows = batch.num_rows();

    let mut builder = Builder::default();
    builder.push_record(columns);
    for i in 0..n_rows {
        let mut row = Vec::with_capacity(col_indices.len() + if row_index { 1 } else { 0 });
        if row_index {
            row.push((row_index_start + i).to_string());
        }
        for (col_idx, field) in &col_indices {
            let col = batch.column(*col_idx);
            let s = format_cell(
                col,
                i,
                field.data_type(),
                Some(field.name()),
                time_format,
                precision,
            )?;
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
                    let suffix =
                        if j < element_name_parts.len() && !element_name_parts[j].is_empty() {
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
    let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut out, batch.schema_ref())
        .into_diagnostic()?;
    writer.write(batch).into_diagnostic()?;
    writer.finish().into_diagnostic()?;
    out.flush().into_diagnostic()?;
    Ok(())
}

#[cfg(feature = "parquet")]
/// Prints a RecordBatch as Parquet to stdout.
fn print_record_batch_parquet(batch: &RecordBatch) -> miette::Result<()> {
    let mut out = std::io::stdout();
    let mut writer =
        parquet::arrow::ArrowWriter::try_new(&mut out, batch.schema_ref().clone(), None)
            .into_diagnostic()?;
    writer.write(batch).into_diagnostic()?;
    writer.close().into_diagnostic()?;
    out.flush().into_diagnostic()?;
    Ok(())
}

/// Prints a RecordBatch as CSV to stdout (same cell formatting as table).
fn print_record_batch_csv(
    batch: &RecordBatch,
    time_format: TimeFormat,
    precision: &Precision,
    row_index: bool,
    row_index_start: usize,
) -> miette::Result<()> {
    let schema = batch.schema();
    let col_indices: Vec<(usize, &Arc<Field>)> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| time_format != TimeFormat::Omit || f.name() != "time")
        .collect();
    let n_rows = batch.num_rows();
    let mut out = std::io::stdout();

    if row_index {
        write!(out, "{}", csv_escape("index")).into_diagnostic()?;
        if !col_indices.is_empty() {
            write!(out, ",").into_diagnostic()?;
        }
    }
    for (j, (_, field)) in col_indices.iter().enumerate() {
        if j > 0 {
            write!(out, ",").into_diagnostic()?;
        }
        let header = column_header_with_unit(field.name(), time_format);
        write!(out, "{}", csv_escape(&header)).into_diagnostic()?;
    }
    writeln!(out).into_diagnostic()?;

    for i in 0..n_rows {
        if row_index {
            write!(out, "{}", row_index_start + i).into_diagnostic()?;
            if !col_indices.is_empty() {
                write!(out, ",").into_diagnostic()?;
            }
        }
        for (j, (col_idx, field)) in col_indices.iter().enumerate() {
            if j > 0 {
                write!(out, ",").into_diagnostic()?;
            }
            let col = batch.column(*col_idx);
            let s = format_cell(
                col,
                i,
                field.data_type(),
                Some(field.name()),
                time_format,
                precision,
            )?;
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
    column_name: Option<&str>,
    time_format: TimeFormat,
    precision: &Precision,
) -> miette::Result<String> {
    use arrow::array::*;
    use arrow::datatypes::DataType;
    use impeller2::types::Timestamp;

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
            let x = a.value(row);
            match precision {
                Precision::Full => format!("{}", x),
                Precision::Decimals(n) => format!("{:.1$}", x, *n as usize),
            }
        }
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
            let x = a.value(row);
            match precision {
                Precision::Full => format!("{}", x),
                Precision::Decimals(n) => format!("{:.1$}", x, *n as usize),
            }
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
            let value_us = if let Some(a) = col
                .as_any()
                .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
            {
                a.value(row)
            } else if let Some(a) = col
                .as_any()
                .downcast_ref::<arrow::array::TimestampMillisecondArray>()
            {
                a.value(row).saturating_mul(1000)
            } else if let Some(a) = col
                .as_any()
                .downcast_ref::<arrow::array::TimestampNanosecondArray>()
            {
                a.value(row) / 1000
            } else {
                return Ok("<ts>".to_string());
            };
            match (column_name == Some("time"), time_format) {
                (true, TimeFormat::Omit) => String::new(),
                (true, TimeFormat::Datetime) => {
                    format!("{}", hifitime::Epoch::from(Timestamp(value_us)))
                }
                (true, TimeFormat::Seconds) => {
                    format!("{}", value_us as f64 / 1_000_000.0)
                }
                (true, TimeFormat::Milliseconds) => {
                    format!("{}", value_us as f64 / 1_000.0)
                }
                (true, TimeFormat::Microseconds) => format!("{}", value_us),
                (false, _) => format!("{}", hifitime::Epoch::from(Timestamp(value_us))),
            }
        }
        DataType::FixedSizeList(_, _) | DataType::List(_) => {
            let a = col.as_any().downcast_ref::<FixedSizeListArray>();
            if let Some(arr) = a {
                let vals = arr.value(row);
                let len = vals.len();
                let parts: Vec<String> = (0..len)
                    .map(|j| format_cell(&vals, j, vals.data_type(), None, time_format, precision))
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
