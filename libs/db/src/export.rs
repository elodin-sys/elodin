//! Database export tool
//!
//! Exports database contents to parquet, arrow-ipc, or csv files without requiring a running server.

use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufWriter, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// 1 MiB output buffer — large enough that even multi-GB CSVs flush only a few thousand
/// times instead of millions of small writes (default `csv::Writer` buffer is ~8 KiB).
const FILE_BUF_CAP: usize = 1 << 20;

use arrow::array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Int32Array, Int32Builder, Int64Array,
    LargeStringBuilder, PrimitiveArray, RecordBatch, TimestampMicrosecondArray,
};
use arrow::compute;
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Field, FieldRef, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, Schema, TimeUnit, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use glob::Pattern;
use tracing::{info_span, trace_span};

use crate::cancellation::check_cancelled;
use crate::{Component, DB, Error};

/// Export format for the CLI command.
/// This is separate from the internal ArchiveFormat to exclude "native".
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ExportFormat {
    Parquet,
    ArrowIpc,
    Csv,
}

/// Build the JSON-style "[v0, v1, ...]" rows for a FixedSizeList over a primitive array,
/// reusing one `String` buffer across all rows. Display semantics are preserved exactly so
/// the output is byte-identical to the previous `format!("{}", v)`-per-cell implementation.
///
/// `fmt_value` formats one element of the primitive's `Native` type into the row buffer.
/// Pulled out as a closure so the same routine is reused by the (default) Display path
/// and the opt-in fast-floats path (Phase 3d).
fn build_list_strings_primitive<T, F>(
    array: &FixedSizeListArray,
    list_size: usize,
    builder: &mut LargeStringBuilder,
    fmt_value: F,
) where
    T: ArrowPrimitiveType,
    F: Fn(T::Native, &mut String),
{
    let values_arr = array
        .values()
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .expect("FixedSizeList values primitive type mismatch");
    let raw: &[T::Native] = values_arr.values();
    let has_nulls = values_arr.null_count() > 0;
    let mut row = String::with_capacity(16 + 12 * list_size);

    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null();
            continue;
        }
        row.clear();
        row.push('[');
        let base = i * list_size;
        for j in 0..list_size {
            if j != 0 {
                row.push_str(", ");
            }
            let idx = base + j;
            if has_nulls && values_arr.is_null(idx) {
                row.push_str("null");
            } else {
                fmt_value(raw[idx], &mut row);
            }
        }
        row.push(']');
        builder.append_value(&row);
    }
}

/// Boolean variant of [`build_list_strings_primitive`] (BooleanArray is bit-packed and not
/// a `PrimitiveArray`).
fn build_list_strings_bool(
    array: &FixedSizeListArray,
    list_size: usize,
    builder: &mut LargeStringBuilder,
) {
    let values_arr = array
        .values()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("FixedSizeList values BooleanArray mismatch");
    let has_nulls = values_arr.null_count() > 0;
    let mut row = String::with_capacity(16 + 7 * list_size);

    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null();
            continue;
        }
        row.clear();
        row.push('[');
        let base = i * list_size;
        for j in 0..list_size {
            if j != 0 {
                row.push_str(", ");
            }
            let idx = base + j;
            if has_nulls && values_arr.is_null(idx) {
                row.push_str("null");
            } else if values_arr.value(idx) {
                row.push_str("true");
            } else {
                row.push_str("false");
            }
        }
        row.push(']');
        builder.append_value(&row);
    }
}

/// Convert a FixedSizeListArray to a LargeStringArray with JSON-like representation.
/// Each array element becomes a string like "[1.0, 2.0, 3.0]".
/// Uses LargeStringBuilder (i64 offsets) to handle very large arrays that would overflow
/// StringBuilder's i32 offset limit (~2GB).
///
/// The element-type dispatch happens **once** at the top of the function; the inner loop
/// is monomorphized per primitive type and reuses a single `String` buffer (no per-cell
/// allocation, no per-row `Vec`, no `parts.join`).
///
/// When `fast_floats` is false (default) the f32/f64 cells are written via `std::fmt::Display`
/// — byte-identical to the previous `format!("{}", v)`-per-cell output. When `fast_floats`
/// is true, f32/f64 cells go through `ryu`, which is markedly faster but produces a slightly
/// different (still parseable, round-trippable) text format.
fn fixed_size_list_to_string(array: &FixedSizeListArray, fast_floats: bool) -> ArrayRef {
    let list_size = array.value_length() as usize;
    let _span = trace_span!(
        "fixed_size_list_to_string",
        rows = array.len(),
        list_size,
        fast_floats,
    )
    .entered();
    let mut builder = LargeStringBuilder::new();

    // Check element type once before entering the loop.
    // Arrow arrays are homogeneously typed, so we can determine the type from
    // the FixedSizeList's value type in the schema.
    let inner_field = match array.data_type() {
        DataType::FixedSizeList(field, _) => field,
        _ => unreachable!("FixedSizeListArray must have FixedSizeList data type"),
    };

    // Closures that format one Native value into a row buffer.
    // `write!(..., "{v}")` writes into the existing String without allocating; this preserves
    // the exact `format!("{}", v)` byte output. The integer/bool variants are always Display.
    let fmt_i64 = |v: i64, buf: &mut String| {
        write!(buf, "{v}").unwrap();
    };
    let fmt_i32 = |v: i32, buf: &mut String| {
        write!(buf, "{v}").unwrap();
    };
    let fmt_i16 = |v: i16, buf: &mut String| {
        write!(buf, "{v}").unwrap();
    };
    let fmt_i8 = |v: i8, buf: &mut String| {
        write!(buf, "{v}").unwrap();
    };
    let fmt_u64 = |v: u64, buf: &mut String| {
        write!(buf, "{v}").unwrap();
    };
    let fmt_u32 = |v: u32, buf: &mut String| {
        write!(buf, "{v}").unwrap();
    };
    let fmt_u16 = |v: u16, buf: &mut String| {
        write!(buf, "{v}").unwrap();
    };
    let fmt_u8 = |v: u8, buf: &mut String| {
        write!(buf, "{v}").unwrap();
    };

    match inner_field.data_type() {
        DataType::Float64 => {
            if fast_floats {
                build_list_strings_primitive::<Float64Type, _>(
                    array,
                    list_size,
                    &mut builder,
                    write_f64_fast,
                )
            } else {
                build_list_strings_primitive::<Float64Type, _>(
                    array,
                    list_size,
                    &mut builder,
                    |v: f64, buf: &mut String| write!(buf, "{v}").unwrap(),
                )
            }
        }
        DataType::Float32 => {
            if fast_floats {
                build_list_strings_primitive::<Float32Type, _>(
                    array,
                    list_size,
                    &mut builder,
                    write_f32_fast,
                )
            } else {
                build_list_strings_primitive::<Float32Type, _>(
                    array,
                    list_size,
                    &mut builder,
                    |v: f32, buf: &mut String| write!(buf, "{v}").unwrap(),
                )
            }
        }
        DataType::Int64 => {
            build_list_strings_primitive::<Int64Type, _>(array, list_size, &mut builder, fmt_i64)
        }
        DataType::Int32 => {
            build_list_strings_primitive::<Int32Type, _>(array, list_size, &mut builder, fmt_i32)
        }
        DataType::Int16 => {
            build_list_strings_primitive::<Int16Type, _>(array, list_size, &mut builder, fmt_i16)
        }
        DataType::Int8 => {
            build_list_strings_primitive::<Int8Type, _>(array, list_size, &mut builder, fmt_i8)
        }
        DataType::UInt64 => {
            build_list_strings_primitive::<UInt64Type, _>(array, list_size, &mut builder, fmt_u64)
        }
        DataType::UInt32 => {
            build_list_strings_primitive::<UInt32Type, _>(array, list_size, &mut builder, fmt_u32)
        }
        DataType::UInt16 => {
            build_list_strings_primitive::<UInt16Type, _>(array, list_size, &mut builder, fmt_u16)
        }
        DataType::UInt8 => {
            build_list_strings_primitive::<UInt8Type, _>(array, list_size, &mut builder, fmt_u8)
        }
        DataType::Boolean => build_list_strings_bool(array, list_size, &mut builder),
        other => {
            eprintln!(
                "Warning: unsupported data type {:?} in fixed-size list, outputting nulls for all {} rows",
                other,
                array.len()
            );
            // Build a null-list representation once; reuse it for every non-null row.
            let mut null_list = String::with_capacity(16 + 6 * list_size);
            null_list.push('[');
            for j in 0..list_size {
                if j != 0 {
                    null_list.push_str(", ");
                }
                null_list.push_str("null");
            }
            null_list.push(']');
            for i in 0..array.len() {
                if array.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value(&null_list);
                }
            }
        }
    }

    Arc::new(builder.finish())
}

/// Fast f64 writer: ryu for finite values, manual literals for non-finite (matching Display
/// for NaN/inf so customer parsers don't need a separate code path).
fn write_f64_fast(v: f64, buf: &mut String) {
    if v.is_finite() {
        let mut ryu_buf = ryu::Buffer::new();
        buf.push_str(ryu_buf.format_finite(v));
    } else if v.is_nan() {
        buf.push_str("NaN");
    } else if v > 0.0 {
        buf.push_str("inf");
    } else {
        buf.push_str("-inf");
    }
}

/// Fast f32 writer: ryu for finite values, manual literals for non-finite.
fn write_f32_fast(v: f32, buf: &mut String) {
    if v.is_finite() {
        let mut ryu_buf = ryu::Buffer::new();
        buf.push_str(ryu_buf.format_finite(v));
    } else if v.is_nan() {
        buf.push_str("NaN");
    } else if v > 0.0 {
        buf.push_str("inf");
    } else {
        buf.push_str("-inf");
    }
}

/// Convert a RecordBatch to have FixedSizeList columns represented as strings.
/// This is needed for CSV export since Arrow's CSV writer doesn't support nested types.
fn convert_lists_to_strings(batch: &RecordBatch, fast_floats: bool) -> RecordBatch {
    let _span = trace_span!("convert_lists_to_strings", rows = batch.num_rows()).entered();
    let mut new_fields = Vec::new();
    let mut new_columns = Vec::new();

    for (i, field) in batch.schema().fields().iter().enumerate() {
        let column = batch.column(i);

        match field.data_type() {
            DataType::FixedSizeList(_, _) => {
                // Convert FixedSizeList to String
                let list_array = column
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .unwrap();
                let string_array = fixed_size_list_to_string(list_array, fast_floats);
                new_fields.push(Arc::new(Field::new(
                    field.name(),
                    DataType::LargeUtf8,
                    field.is_nullable(),
                )));
                new_columns.push(string_array);
            }
            _ => {
                // Keep other columns as-is
                new_fields.push(Arc::clone(field));
                new_columns.push(Arc::clone(column));
            }
        }
    }

    let new_schema = Arc::new(Schema::new(new_fields));
    RecordBatch::try_new(new_schema, new_columns).unwrap()
}

/// One component's worth of work collected under the DB read guard. Cheap to clone.
struct ComponentTask {
    component: Component,
    column_name: String,
    element_names: String,
}

/// One member of a group passed to [`build_group_record_batch`].
struct GroupMember {
    component: Component,
    /// The "short" column name that appears in the output (everything after the last `.`
    /// of the original component name when `--join` is on, otherwise the full name).
    short_name: String,
    element_names: String,
}

/// A unit of export work: one or more components that should land in the same output file.
struct ExportGroup {
    /// Output filename stem (sanitized later by [`write_record_batch`]).
    output_name: String,
    members: Vec<GroupMember>,
}

/// Group components by the prefix before the last `.` in their name. Single-member groups
/// (no `.` in the name, or only one component sharing a prefix) export as before.
///
/// Members within a group are deterministically ordered by their short name so column order
/// is stable across runs.
fn group_components_by_prefix(tasks: Vec<ComponentTask>) -> Vec<ExportGroup> {
    let mut by_prefix: BTreeMap<String, Vec<GroupMember>> = BTreeMap::new();
    for task in tasks {
        let (prefix, short) = match task.column_name.rsplit_once('.') {
            Some((p, s)) => (p.to_string(), s.to_string()),
            None => (task.column_name.clone(), task.column_name.clone()),
        };
        by_prefix.entry(prefix).or_default().push(GroupMember {
            component: task.component,
            short_name: short,
            element_names: task.element_names,
        });
    }
    by_prefix
        .into_iter()
        .map(|(prefix, mut members)| {
            members.sort_by(|a, b| a.short_name.cmp(&b.short_name));
            ExportGroup {
                output_name: prefix,
                members,
            }
        })
        .collect()
}

/// Build a single `RecordBatch` for an `ExportGroup`. For one-member groups this is
/// equivalent to the pre-`--join` per-component path. For multi-member groups, members
/// sharing identical timestamps are zipped zero-copy; otherwise a sorted union of
/// timestamps is built and each member is aligned via `compute::take` with NULL fill.
fn build_group_record_batch(group: &ExportGroup, flatten: bool) -> RecordBatch {
    let _span = trace_span!(
        "build_group_record_batch",
        group = %group.output_name,
        members = group.members.len(),
        flatten,
    )
    .entered();

    if group.members.len() == 1 {
        let m = &group.members[0];
        return if flatten {
            m.component
                .as_flat_record_batch(m.short_name.clone(), &m.element_names)
        } else {
            m.component.as_record_batch(m.short_name.clone())
        };
    }

    // Fast path: all members share the same index buffer (byte-equal). Zero-copy zip.
    let first = &group.members[0].component;
    let first_ts_bytes = first.time_series.index().data();
    let identical = group.members.iter().all(|m| {
        std::ptr::eq(&m.component.time_series, &first.time_series)
            || m.component.time_series.index().data() == first_ts_bytes
    });

    if identical {
        join_group_identical_ts(group, flatten)
    } else {
        join_group_outer(group, flatten)
    }
}

/// Fast-path join: every member's index buffer is byte-equal, so we zip member columns
/// onto a shared time axis without any gather work.
fn join_group_identical_ts(group: &ExportGroup, flatten: bool) -> RecordBatch {
    let first = &group.members[0].component;
    let time_array = first.as_time_series_array();
    let len_time = time_array.len();

    let time_field = Arc::new(Field::new(
        "time",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    ));
    let mut fields: Vec<FieldRef> = vec![time_field];
    let mut columns: Vec<ArrayRef> = vec![time_array];

    for member in &group.members {
        if flatten {
            let (mfields, marrays) = member.component.as_flattened_columns(
                member.short_name.clone(),
                ..,
                &member.element_names,
            );
            fields.extend(mfields);
            columns.extend(marrays);
        } else {
            let (f, a) = member.component.as_data_array(member.short_name.clone());
            fields.push(f);
            columns.push(a);
        }
    }

    let len = columns
        .iter()
        .map(|c| c.len())
        .min()
        .unwrap_or(0)
        .min(len_time);
    let columns: Vec<ArrayRef> = columns.into_iter().map(|c| c.slice(0, len)).collect();

    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("record batch params wrong")
}

/// Slow-path join: union all members' timestamp arrays into a sorted axis, then align each
/// member to it via `compute::take` with NULL fill for missing rows.
fn join_group_outer(group: &ExportGroup, flatten: bool) -> RecordBatch {
    // Collect each member's timestamps once.
    let member_ts: Vec<TimestampMicrosecondArray> = group
        .members
        .iter()
        .map(|m| {
            let arr = m.component.as_time_series_array();
            arr.as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("time array type")
                .clone()
        })
        .collect();

    // Sort+dedup union of all timestamps. K-way merge would be faster but this is the
    // fallback path for the rare mismatched-timestamp case so the simpler implementation
    // wins on maintainability.
    let mut all_ts: Vec<i64> = Vec::with_capacity(member_ts.iter().map(|m| m.len()).sum());
    for m in &member_ts {
        all_ts.extend(m.values().iter().copied());
    }
    all_ts.sort_unstable();
    all_ts.dedup();
    let union_ts = TimestampMicrosecondArray::from(all_ts);

    let time_field = Arc::new(Field::new(
        "time",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    ));
    let mut fields: Vec<FieldRef> = vec![time_field];
    let mut columns: Vec<ArrayRef> = vec![Arc::new(union_ts.clone())];

    for (member, m_ts) in group.members.iter().zip(member_ts.iter()) {
        let indices = build_alignment_indices(&union_ts, m_ts);

        // Build a record-batch-equivalent set of columns for this member at its native
        // timestamps, then take() each column to align to the union axis. Note the
        // NULLs in `indices` propagate through take() to produce NULL fills.
        let (member_fields, member_arrays) = if flatten {
            member.component.as_flattened_columns(
                member.short_name.clone(),
                ..,
                &member.element_names,
            )
        } else {
            let (f, a) = member.component.as_data_array(member.short_name.clone());
            (vec![f], vec![a])
        };

        for (f, a) in member_fields.into_iter().zip(member_arrays.into_iter()) {
            // Make the field nullable; take() may produce nulls along the union axis.
            let nullable_field = Arc::new(
                (*f).clone()
                    .with_nullable(true)
                    .with_data_type(a.data_type().clone()),
            );
            let aligned = compute::take(&a, &indices, None).expect("compute::take");
            fields.push(nullable_field);
            columns.push(aligned);
        }
    }

    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("record batch params wrong")
}

/// Build an indices array that maps positions in `union_ts` to positions in `member_ts`,
/// with NULL where the timestamp doesn't appear in `member_ts`. Two-pointer linear merge
/// over already-sorted inputs.
fn build_alignment_indices(
    union_ts: &TimestampMicrosecondArray,
    member_ts: &TimestampMicrosecondArray,
) -> Int32Array {
    let mut builder = Int32Builder::with_capacity(union_ts.len());
    let mut j = 0usize;
    let m_len = member_ts.len();
    for i in 0..union_ts.len() {
        let target = union_ts.value(i);
        while j < m_len && member_ts.value(j) < target {
            j += 1;
        }
        if j < m_len && member_ts.value(j) == target {
            builder.append_value(j as i32);
        } else {
            builder.append_null();
        }
    }
    builder.finish()
}

/// If `time_format` is one of the `Mono*` variants, replace column 0 (the time column,
/// always a `TimestampMicrosecondArray` in our pipeline) with a renamed `Int64Array`
/// carrying integer microseconds or nanoseconds since unix epoch. Default `Iso8601`
/// returns the batch untouched.
fn rewrite_time_column(batch: RecordBatch, time_format: TimeFormat) -> RecordBatch {
    if time_format == TimeFormat::Iso8601 {
        return batch;
    }
    let ts = batch
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .expect("time column at idx 0 must be TimestampMicrosecond");

    let (new_name, int_array): (&'static str, ArrayRef) = match time_format {
        TimeFormat::MonoMicroseconds => (
            "time_us",
            // Zero-copy retype: same i64 buffer, new type tag.
            Arc::new(Int64Array::new(ts.values().clone(), ts.nulls().cloned())),
        ),
        TimeFormat::MonoNanoseconds => {
            // us * 1000 = ns. saturating_mul is defensive — for any timestamp before
            // ~year 2262 this is a regular multiply.
            let scaled: Vec<i64> = ts
                .values()
                .iter()
                .map(|us| us.saturating_mul(1000))
                .collect();
            ("time_ns", Arc::new(Int64Array::from(scaled)))
        }
        TimeFormat::Iso8601 => unreachable!(),
    };

    let mut fields: Vec<FieldRef> = Vec::with_capacity(batch.num_columns());
    fields.push(Arc::new(Field::new(new_name, DataType::Int64, false)));
    fields.extend(batch.schema().fields().iter().skip(1).cloned());

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    columns.push(int_array);
    columns.extend(batch.columns().iter().skip(1).cloned());

    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("record batch params wrong")
}

/// Write a fully-built `RecordBatch` for one group (or one component) to disk in the
/// requested format. Pulled out of `run` so the same code path serves both --join and
/// the single-component default.
fn write_record_batch(
    output_name: &str,
    record_batch: &RecordBatch,
    row_count: usize,
    format: ExportFormat,
    flatten: bool,
    csv_fast_floats: bool,
    output_path: &Path,
) -> Result<(), Error> {
    match format {
        ExportFormat::ArrowIpc => {
            let schema = record_batch.schema_ref();
            let file_name = format!("{output_name}.arrow");
            let file_path = output_path.join(&file_name);
            let file = File::create(&file_path)?;
            let mut buf = BufWriter::with_capacity(FILE_BUF_CAP, file);
            let mut writer = arrow::ipc::writer::FileWriter::try_new(&mut buf, schema)?;
            let _w_span = trace_span!("write_arrow_ipc", rows = row_count).entered();
            writer.write(record_batch)?;
            writer.finish()?;
            drop(writer);
            buf.flush()?;
            drop(_w_span);
            println!("  Exported {} ({} rows)", file_name, row_count);
        }
        #[cfg(feature = "parquet")]
        ExportFormat::Parquet => {
            let schema = record_batch.schema_ref();
            let file_name = format!("{output_name}.parquet");
            let file_path = output_path.join(&file_name);
            let file = File::create(&file_path)?;
            let mut buf = BufWriter::with_capacity(FILE_BUF_CAP, file);
            let mut writer = parquet::arrow::ArrowWriter::try_new(&mut buf, schema.clone(), None)?;
            let _w_span = trace_span!("write_parquet", rows = row_count).entered();
            writer.write(record_batch)?;
            writer.close()?;
            buf.flush()?;
            drop(_w_span);
            println!("  Exported {} ({} rows)", file_name, row_count);
        }
        #[cfg(not(feature = "parquet"))]
        ExportFormat::Parquet => {
            return Err(Error::UnsupportedArchiveFormat);
        }
        ExportFormat::Csv => {
            let csv_batch = if flatten {
                record_batch.clone()
            } else {
                convert_lists_to_strings(record_batch, csv_fast_floats)
            };
            let file_name = format!("{output_name}.csv");
            let file_path = output_path.join(&file_name);
            let file = File::create(&file_path)?;
            let mut buf = BufWriter::with_capacity(FILE_BUF_CAP, file);
            let mut writer = arrow::csv::Writer::new(&mut buf);
            {
                let _w_span = trace_span!("write_arrow_csv", rows = row_count).entered();
                writer.write(&csv_batch)?;
            }
            drop(writer);
            buf.flush()?;
            println!("  Exported {} ({} rows)", file_name, row_count);
        }
    }
    Ok(())
}

/// How to render the time column in the output.
///
/// `Iso8601` (the default) preserves historical behaviour: a `Timestamp(Microsecond)`
/// column named `time` that arrow's CSV writer renders as ISO 8601. The two `Mono*`
/// variants replace the column with a renamed `Int64` column carrying the integer
/// value directly, so it can be diffed against customer-provided monotonic timestamps
/// (e.g. a `TIME_MONOTONIC` field) without any text-time conversion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimeFormat {
    /// Default: `Timestamp(Microsecond)` column named `time`. Arrow writes ISO 8601.
    #[default]
    Iso8601,
    /// `Int64` column named `time_ns` carrying microseconds-since-epoch * 1000.
    MonoNanoseconds,
    /// `Int64` column named `time_us` carrying microseconds-since-epoch as-is.
    MonoMicroseconds,
}

/// Optional flags that influence how `export::run` writes output.
///
/// Built up by the CLI in [`crate::main`] from `ExportArgs`. Defaults match the historical
/// behaviour of `export::run` (no flatten, no pattern, std `Display` floats, no joining,
/// ISO 8601 time column).
#[derive(Clone, Debug, Default)]
pub struct ExportOptions {
    /// Flatten FixedSizeList columns into one column per element (per-message-field layout).
    pub flatten: bool,
    /// Glob pattern over component names; non-matching components are skipped.
    pub pattern: Option<String>,
    /// CSV-only: use `ryu` for f32/f64 instead of `std::fmt::Display`. Faster but produces
    /// a slightly different (still round-trippable) text format. Off by default.
    pub csv_fast_floats: bool,
    /// Group components by their name prefix (everything before the last `.`) and emit
    /// one file per group. Identical-timestamp components are zipped zero-copy; differing
    /// timestamps are unioned via a sorted merge with NULL fill.
    pub join: bool,
    /// Render the time column as ISO 8601 (default), integer nanoseconds, or integer
    /// microseconds. Applies to all formats (CSV / Parquet / Arrow IPC).
    pub time_format: TimeFormat,
}

/// Export database contents to files.
///
/// # Arguments
/// * `db_path` - Path to the database directory
/// * `output_path` - Output directory for exported files
/// * `format` - Export format (parquet, arrow-ipc, or csv)
/// * `options` - Optional flags (flatten, glob pattern, fast-float formatter, ...).
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(Error)` if the operation fails
pub fn run(
    db_path: PathBuf,
    output_path: PathBuf,
    format: ExportFormat,
    options: ExportOptions,
) -> Result<(), Error> {
    let ExportOptions {
        flatten,
        pattern,
        csv_fast_floats,
        join,
        time_format,
    } = options;

    // Validate database path
    if !db_path.exists() {
        return Err(Error::MissingDbState(db_path));
    }

    let db_state_path = db_path.join("db_state");
    if !db_state_path.exists() {
        return Err(Error::MissingDbState(db_state_path));
    }

    // Parse pattern if provided
    let glob_pattern = pattern
        .as_ref()
        .map(|p| Pattern::new(p))
        .transpose()
        .map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid glob pattern: {}", e),
            ))
        })?;

    println!("Opening database: {}", db_path.display());
    let db = DB::open(db_path)?;

    // Create output directory
    std::fs::create_dir_all(&output_path)?;
    println!("Exporting to: {}", output_path.display());

    let format_name = match format {
        ExportFormat::Parquet => "parquet",
        ExportFormat::ArrowIpc => "arrow-ipc",
        ExportFormat::Csv => "csv",
    };
    println!("Format: {}", format_name);
    if flatten {
        println!("Vector columns will be flattened");
    }
    if join {
        println!("Components will be joined by name prefix (one file per group)");
    }
    if let Some(ref p) = pattern {
        println!("Filter pattern: {}", p);
    }
    println!();

    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // First pass under the read guard: snapshot per-component metadata. Component clones
    // are cheap (Arc-backed TimeSeries). Doing the heavy work outside the read guard lets
    // multiple threads proceed in parallel.
    let (per_component, pre_skipped): (Vec<ComponentTask>, usize) = db.with_state(|state| {
        let total_components = state.components.len();
        println!("Found {} components", total_components);

        let mut work = Vec::with_capacity(total_components);
        let mut pre_skipped = 0usize;
        for component in state.components.values() {
            let Some(component_metadata) = state.component_metadata.get(&component.component_id)
            else {
                pre_skipped += 1;
                continue;
            };
            let column_name = component_metadata.name.clone();
            if let Some(ref pattern) = glob_pattern
                && !pattern.matches(&column_name)
            {
                pre_skipped += 1;
                continue;
            }
            let element_names = component_metadata.element_names().to_string();
            work.push(ComponentTask {
                component: component.clone(),
                column_name,
                element_names,
            });
        }
        (work, pre_skipped)
    });

    // If --join is set, group by the prefix before the last '.'; otherwise each component
    // becomes its own one-member group (i.e. existing per-component behaviour).
    let groups: Vec<ExportGroup> = if join {
        group_components_by_prefix(per_component)
    } else {
        per_component
            .into_iter()
            .map(|task| ExportGroup {
                output_name: task.column_name.clone(),
                members: vec![GroupMember {
                    component: task.component,
                    short_name: task.column_name,
                    element_names: task.element_names,
                }],
            })
            .collect()
    };

    let exported_count = AtomicUsize::new(0);
    let skipped_count = AtomicUsize::new(pre_skipped);

    groups
        .par_iter()
        .try_for_each(|group| -> Result<(), Error> {
            check_cancelled()?;
            let _g_span = info_span!(
                "export_group",
                group = %group.output_name,
                members = group.members.len(),
                flatten,
                join,
            )
            .entered();

            let record_batch = build_group_record_batch(group, flatten);
            let record_batch = rewrite_time_column(record_batch, time_format);
            let row_count = record_batch.num_rows();

            if row_count == 0 {
                println!("  Skipping {} (empty)", group.output_name);
                skipped_count.fetch_add(group.members.len(), Ordering::Relaxed);
                return Ok(());
            }

            write_record_batch(
                &group.output_name,
                &record_batch,
                row_count,
                format,
                flatten,
                csv_fast_floats,
                &output_path,
            )?;

            exported_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })?;

    let exported_count = exported_count.load(Ordering::Relaxed);
    let skipped_count = skipped_count.load(Ordering::Relaxed);

    println!();
    println!(
        "Export complete: {} components exported, {} skipped",
        exported_count, skipped_count
    );

    // Flush stdout to ensure all output is visible
    std::io::stdout().flush()?;

    Ok(())
}
