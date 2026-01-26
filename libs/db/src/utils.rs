//! Shared utility functions for database operations.

use std::path::Path;

use crate::MetadataExt;
use impeller2::types::PrimType;
use impeller2_wkt::ComponentMetadata;

/// Read a primitive value from a byte slice at the given offset and convert it to f64.
///
/// This is useful for converting time series data of various primitive types
/// into a common f64 format for visualization and downsampling.
///
/// # Arguments
/// * `prim_type` - The primitive type of the value
/// * `data` - The byte slice containing the data
/// * `offset` - The byte offset to read from
///
/// # Returns
/// The value as f64, or f64::NAN if the offset is out of bounds
pub fn read_prim_as_f64(prim_type: PrimType, data: &[u8], offset: usize) -> f64 {
    match prim_type {
        PrimType::F64 => {
            let bytes: [u8; 8] = data
                .get(offset..offset + 8)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 8]);
            f64::from_le_bytes(bytes)
        }
        PrimType::F32 => {
            let bytes: [u8; 4] = data
                .get(offset..offset + 4)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 4]);
            f32::from_le_bytes(bytes) as f64
        }
        PrimType::U64 => {
            let bytes: [u8; 8] = data
                .get(offset..offset + 8)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 8]);
            u64::from_le_bytes(bytes) as f64
        }
        PrimType::U32 => {
            let bytes: [u8; 4] = data
                .get(offset..offset + 4)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 4]);
            u32::from_le_bytes(bytes) as f64
        }
        PrimType::U16 => {
            let bytes: [u8; 2] = data
                .get(offset..offset + 2)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 2]);
            u16::from_le_bytes(bytes) as f64
        }
        PrimType::U8 => data.get(offset).copied().unwrap_or(0) as f64,
        PrimType::I64 => {
            let bytes: [u8; 8] = data
                .get(offset..offset + 8)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 8]);
            i64::from_le_bytes(bytes) as f64
        }
        PrimType::I32 => {
            let bytes: [u8; 4] = data
                .get(offset..offset + 4)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 4]);
            i32::from_le_bytes(bytes) as f64
        }
        PrimType::I16 => {
            let bytes: [u8; 2] = data
                .get(offset..offset + 2)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 2]);
            i16::from_le_bytes(bytes) as f64
        }
        PrimType::I8 => data.get(offset).copied().unwrap_or(0) as i8 as f64,
        PrimType::Bool => {
            if data.get(offset).copied().unwrap_or(0) != 0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Get a human-readable label for a component path.
///
/// Returns the component name from metadata if available, formatted as
/// "name (id)" if they differ, or just the directory name as a fallback.
pub fn component_label(path: &Path) -> String {
    let fallback = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<unknown>")
        .to_string();

    let metadata_path = path.join("metadata");
    let name = if metadata_path.exists() {
        read_component_name(&metadata_path).unwrap_or_else(|_| fallback.clone())
    } else {
        fallback.clone()
    };

    if name == fallback {
        name
    } else {
        format!("{name} ({fallback})")
    }
}

/// Read a component's name from its metadata file.
pub fn read_component_name(metadata_path: &Path) -> Result<String, std::io::Error> {
    let metadata = <ComponentMetadata as MetadataExt>::read(metadata_path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(metadata.name)
}
