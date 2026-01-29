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

/// Convert a FixedSizeListArray to a LargeStringArray with JSON-like representation.
/// Each array element becomes a string like "[1.0, 2.0, 3.0]".
/// Uses LargeStringBuilder (i64 offsets) to handle very large arrays that would overflow
/// StringBuilder's i32 offset limit (~2GB).
fn fixed_size_list_to_string(array: &FixedSizeListArray) -> ArrayRef {
    let mut builder = LargeStringBuilder::new();
    let list_size = array.value_length() as usize;

    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null();
        } else {
            let values = array.value(i);
            let mut parts = Vec::with_capacity(list_size);

            for j in 0..list_size {
                // Convert each element to string based on its type
                let value_str = match values.data_type() {
                    DataType::Float64 => {
                        let arr = values
                            .as_any()
                            .downcast_ref::<arrow::array::Float64Array>()
                            .unwrap();
                        format!("{}", arr.value(j))
                    }
                    DataType::Float32 => {
                        let arr = values
                            .as_any()
                            .downcast_ref::<arrow::array::Float32Array>()
                            .unwrap();
                        format!("{}", arr.value(j))
                    }
                    DataType::Int64 => {
                        let arr = values
                            .as_any()
                            .downcast_ref::<arrow::array::Int64Array>()
                            .unwrap();
                        format!("{}", arr.value(j))
                    }
                    DataType::Int32 => {
                        let arr = values
                            .as_any()
                            .downcast_ref::<arrow::array::Int32Array>()
                            .unwrap();
                        format!("{}", arr.value(j))
                    }
                    DataType::Int16 => {
                        let arr = values
                            .as_any()
                            .downcast_ref::<arrow::array::Int16Array>()
                            .unwrap();
                        format!("{}", arr.value(j))
                    }
                    DataType::Int8 => {
                        let arr = values
                            .as_any()
                            .downcast_ref::<arrow::array::Int8Array>()
                            .unwrap();
                        format!("{}", arr.value(j))
                    }
                    DataType::UInt64 => {
                        let arr = values
                            .as_any()
                            .downcast_ref::<arrow::array::UInt64Array>()
                            .unwrap();
                        format!("{}", arr.value(j))
                    }
                    DataType::UInt32 => {
                        let arr = values
                            .as_any()
                            .downcast_ref::<arrow::array::UInt32Array>()
                            .unwrap();
                        format!("{}", arr.value(j))
                    }
                    DataType::UInt16 => {
                        let arr = values
                            .as_any()
                            .downcast_ref::<arrow::array::UInt16Array>()
                            .unwrap();
                        format!("{}", arr.value(j))
                    }
                    DataType::UInt8 => {
                        let arr = values
                            .as_any()
                            .downcast_ref::<arrow::array::UInt8Array>()
                            .unwrap();
                        format!("{}", arr.value(j))
                    }
                    DataType::Boolean => {
                        let arr = values
                            .as_any()
                            .downcast_ref::<arrow::array::BooleanArray>()
                            .unwrap();
                        format!("{}", arr.value(j))
                    }
                    _ => "null".to_string(),
                };
                parts.push(value_str);
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
