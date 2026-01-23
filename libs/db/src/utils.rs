//! Shared utility functions for database operations.

use std::path::Path;

use crate::MetadataExt;
use impeller2_wkt::ComponentMetadata;

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
