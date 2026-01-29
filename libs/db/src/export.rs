//! Database export tool
//!
//! Exports database contents to parquet, arrow-ipc, or csv files without requiring a running server.

use std::fs::File;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeListArray, LargeStringBuilder, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use glob::Pattern;

use crate::cancellation::check_cancelled;
use crate::{DB, Error};

/// Export format for the CLI command.
/// This is separate from the internal ArchiveFormat to exclude "native".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportFormat {
    Parquet,
    ArrowIpc,
    Csv,
}

/// Supported element types for fixed-size list to string conversion.
#[derive(Clone, Copy)]
enum ListElementType {
    Float64,
    Float32,
    Int64,
    Int32,
    Int16,
    Int8,
    UInt64,
    UInt32,
    UInt16,
    UInt8,
    Boolean,
}

impl ListElementType {
    /// Try to determine the element type from an Arrow DataType.
    /// Returns None for unsupported types.
    fn from_arrow(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Float64 => Some(Self::Float64),
            DataType::Float32 => Some(Self::Float32),
            DataType::Int64 => Some(Self::Int64),
            DataType::Int32 => Some(Self::Int32),
            DataType::Int16 => Some(Self::Int16),
            DataType::Int8 => Some(Self::Int8),
            DataType::UInt64 => Some(Self::UInt64),
            DataType::UInt32 => Some(Self::UInt32),
            DataType::UInt16 => Some(Self::UInt16),
            DataType::UInt8 => Some(Self::UInt8),
            DataType::Boolean => Some(Self::Boolean),
            _ => None,
        }
    }

    /// Format a single element at index `j` from the given array.
    fn format_element(&self, values: &dyn Array, j: usize) -> String {
        match self {
            Self::Float64 => {
                let arr = values
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();
                format!("{}", arr.value(j))
            }
            Self::Float32 => {
                let arr = values
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .unwrap();
                format!("{}", arr.value(j))
            }
            Self::Int64 => {
                let arr = values
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .unwrap();
                format!("{}", arr.value(j))
            }
            Self::Int32 => {
                let arr = values
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .unwrap();
                format!("{}", arr.value(j))
            }
            Self::Int16 => {
                let arr = values
                    .as_any()
                    .downcast_ref::<arrow::array::Int16Array>()
                    .unwrap();
                format!("{}", arr.value(j))
            }
            Self::Int8 => {
                let arr = values
                    .as_any()
                    .downcast_ref::<arrow::array::Int8Array>()
                    .unwrap();
                format!("{}", arr.value(j))
            }
            Self::UInt64 => {
                let arr = values
                    .as_any()
                    .downcast_ref::<arrow::array::UInt64Array>()
                    .unwrap();
                format!("{}", arr.value(j))
            }
            Self::UInt32 => {
                let arr = values
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .unwrap();
                format!("{}", arr.value(j))
            }
            Self::UInt16 => {
                let arr = values
                    .as_any()
                    .downcast_ref::<arrow::array::UInt16Array>()
                    .unwrap();
                format!("{}", arr.value(j))
            }
            Self::UInt8 => {
                let arr = values
                    .as_any()
                    .downcast_ref::<arrow::array::UInt8Array>()
                    .unwrap();
                format!("{}", arr.value(j))
            }
            Self::Boolean => {
                let arr = values
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .unwrap();
                format!("{}", arr.value(j))
            }
        }
    }
}

/// Convert a FixedSizeListArray to a LargeStringArray with JSON-like representation.
/// Each array element becomes a string like "[1.0, 2.0, 3.0]".
/// Uses LargeStringBuilder (i64 offsets) to handle very large arrays that would overflow
/// StringBuilder's i32 offset limit (~2GB).
fn fixed_size_list_to_string(array: &FixedSizeListArray) -> ArrayRef {
    let mut builder = LargeStringBuilder::new();
    let list_size = array.value_length() as usize;

    // Check element type once before entering the loop.
    // Arrow arrays are homogeneously typed, so we can determine the type from
    // the FixedSizeList's value type in the schema.
    let inner_field = match array.data_type() {
        DataType::FixedSizeList(field, _) => field,
        _ => unreachable!("FixedSizeListArray must have FixedSizeList data type"),
    };
    let element_type = ListElementType::from_arrow(inner_field.data_type());

    // If unsupported type, warn once and return array of null-filled strings
    let Some(element_type) = element_type else {
        eprintln!(
            "Warning: unsupported data type {:?} in fixed-size list, outputting nulls for all {} rows",
            inner_field.data_type(),
            array.len()
        );

        // Build "[null, null, ...]" string once
        let null_list = format!("[{}]", vec!["null"; list_size].join(", "));
        for i in 0..array.len() {
            if array.is_null(i) {
                builder.append_null();
            } else {
                builder.append_value(&null_list);
            }
        }
        return Arc::new(builder.finish());
    };

    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null();
        } else {
            let values = array.value(i);
            let mut parts = Vec::with_capacity(list_size);

            for j in 0..list_size {
                parts.push(element_type.format_element(values.as_ref(), j));
            }

            builder.append_value(format!("[{}]", parts.join(", ")));
        }
    }

    Arc::new(builder.finish())
}

/// Convert a RecordBatch to have FixedSizeList columns represented as strings.
/// This is needed for CSV export since Arrow's CSV writer doesn't support nested types.
fn convert_lists_to_strings(batch: &RecordBatch) -> RecordBatch {
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
                let string_array = fixed_size_list_to_string(list_array);
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

/// Export database contents to files.
///
/// # Arguments
/// * `db_path` - Path to the database directory
/// * `output_path` - Output directory for exported files
/// * `format` - Export format (parquet, arrow-ipc, or csv)
/// * `flatten` - If true, flatten vector columns to separate columns
/// * `pattern` - Optional glob pattern to filter components by name
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(Error)` if the operation fails
pub fn run(
    db_path: PathBuf,
    output_path: PathBuf,
    format: ExportFormat,
    flatten: bool,
    pattern: Option<String>,
) -> Result<(), Error> {
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
    if let Some(ref p) = pattern {
        println!("Filter pattern: {}", p);
    }
    println!();

    // Export components
    let mut exported_count = 0;
    let mut skipped_count = 0;

    db.with_state(|state| -> Result<(), Error> {
        let total_components = state.components.len();
        println!("Found {} components", total_components);

        for component in state.components.values() {
            // Check for cancellation at the start of each component
            check_cancelled()?;

            let Some(component_metadata) = state.component_metadata.get(&component.component_id)
            else {
                skipped_count += 1;
                continue;
            };

            let column_name = component_metadata.name.clone();

            // Apply glob pattern filter if provided
            if let Some(ref pattern) = glob_pattern
                && !pattern.matches(&column_name)
            {
                skipped_count += 1;
                continue;
            }

            let element_names = component_metadata.element_names();

            // Use flattened or non-flattened record batch based on flag
            let record_batch = if flatten {
                component.as_flat_record_batch(column_name.clone(), element_names)
            } else {
                component.as_record_batch(column_name.clone())
            };

            // Check if component has any data
            let row_count = record_batch.num_rows();
            if row_count == 0 {
                println!("  Skipping {} (empty)", column_name);
                skipped_count += 1;
                continue;
            }

            match format {
                ExportFormat::ArrowIpc => {
                    let schema = record_batch.schema_ref();
                    let file_name = format!("{column_name}.arrow");
                    let file_path = output_path.join(&file_name);
                    let mut file = File::create(&file_path)?;
                    let mut writer = arrow::ipc::writer::FileWriter::try_new(&mut file, schema)?;
                    writer.write(&record_batch)?;
                    writer.finish()?;
                    println!("  Exported {} ({} rows)", file_name, row_count);
                }
                #[cfg(feature = "parquet")]
                ExportFormat::Parquet => {
                    let schema = record_batch.schema_ref();
                    let file_name = format!("{column_name}.parquet");
                    let file_path = output_path.join(&file_name);
                    let mut file = File::create(&file_path)?;
                    let mut writer =
                        parquet::arrow::ArrowWriter::try_new(&mut file, schema.clone(), None)?;
                    writer.write(&record_batch)?;
                    writer.close()?;
                    println!("  Exported {} ({} rows)", file_name, row_count);
                }
                #[cfg(not(feature = "parquet"))]
                ExportFormat::Parquet => {
                    return Err(Error::UnsupportedArchiveFormat);
                }
                ExportFormat::Csv => {
                    // For CSV, convert FixedSizeList columns to strings if not flattening
                    let csv_batch = if flatten {
                        record_batch
                    } else {
                        convert_lists_to_strings(&record_batch)
                    };
                    let file_name = format!("{column_name}.csv");
                    let file_path = output_path.join(&file_name);
                    let mut file = File::create(&file_path)?;
                    let mut writer = arrow::csv::Writer::new(&mut file);
                    writer.write(&csv_batch)?;
                    println!("  Exported {} ({} rows)", file_name, row_count);
                }
            }

            exported_count += 1;
        }
        Ok(())
    })?;

    println!();
    println!(
        "Export complete: {} components exported, {} skipped",
        exported_count, skipped_count
    );

    // Flush stdout to ensure all output is visible
    std::io::stdout().flush()?;

    Ok(())
}
