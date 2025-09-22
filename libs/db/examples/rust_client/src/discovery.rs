use anyhow::{Context, Result};
use colored::*;
use impeller2::schema::Schema;
use impeller2::types::{ComponentId, PrimType};
use impeller2_stellar::Client;
use impeller2_wkt::{DumpMetadata, DumpMetadataResp, DumpSchema, DumpSchemaResp};
use std::collections::HashMap;
use tracing::{debug, info};

/// Component information discovered from the database
#[derive(Debug, Clone)]
pub struct DiscoveredComponent {
    pub _id: ComponentId, // Kept for potential future use
    pub name: String,
    pub schema: Schema<Vec<u64>>,
    pub metadata: HashMap<String, String>,
}

/// Discovers all components registered in the database
pub async fn discover_components(
    client: &mut Client,
) -> Result<HashMap<ComponentId, DiscoveredComponent>> {
    info!("Discovering components from database");

    // Request all metadata
    let dump_metadata = DumpMetadata;
    let metadata_resp: DumpMetadataResp = client
        .request(&dump_metadata)
        .await
        .context("Failed to dump metadata")?;

    println!("\n📊 Discovered Components:");
    println!(
        "  Found {} components registered",
        metadata_resp.component_metadata.len()
    );

    // Request all schemas
    let dump_schema = DumpSchema;
    let schema_resp: DumpSchemaResp = client
        .request(&dump_schema)
        .await
        .context("Failed to dump schemas")?;

    // Build component map
    let mut components = HashMap::new();

    for metadata in metadata_resp.component_metadata {
        // Find matching schema
        if let Some(schema) = schema_resp
            .schemas
            .iter()
            .find(|(id, _)| **id == metadata.component_id)
            .map(|(_, s)| s.clone())
        {
            let component = DiscoveredComponent {
                _id: metadata.component_id,
                name: metadata.name.clone(),
                schema,
                metadata: metadata.metadata.clone(),
            };

            // Display component info
            print_component_info(&component);

            components.insert(metadata.component_id, component);
        }
    }

    debug!("Discovered {} components with schemas", components.len());
    Ok(components)
}

/// Pretty print component information
fn print_component_info(component: &DiscoveredComponent) {
    let schema_str = format_schema(&component.schema);
    println!(
        "  {} {} {} {}",
        "✓".green(),
        component.name.cyan().bold(),
        "→".dimmed(),
        schema_str.yellow()
    );

    if !component.metadata.is_empty() {
        for (key, value) in &component.metadata {
            println!("      {} = {}", key.dimmed(), value);
        }
    }
}

/// Format a schema for display
fn format_schema(schema: &Schema<Vec<u64>>) -> String {
    let prim_type_str = format_prim_type(schema.prim_type());
    let shape = schema.dim();

    if shape.is_empty() {
        prim_type_str.to_string()
    } else {
        format!(
            "{}[{}]",
            prim_type_str,
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Format primitive type for display
fn format_prim_type(prim_type: PrimType) -> &'static str {
    match prim_type {
        PrimType::U8 => "u8",
        PrimType::U16 => "u16",
        PrimType::U32 => "u32",
        PrimType::U64 => "u64",
        PrimType::I8 => "i8",
        PrimType::I16 => "i16",
        PrimType::I32 => "i32",
        PrimType::I64 => "i64",
        PrimType::Bool => "bool",
        PrimType::F32 => "f32",
        PrimType::F64 => "f64",
    }
}

/// Subscribe to rocket-specific components
#[allow(dead_code)]
pub fn filter_rocket_components(
    components: &HashMap<ComponentId, DiscoveredComponent>,
) -> Vec<&DiscoveredComponent> {
    components
        .values()
        .filter(|c| c.name.starts_with("rocket."))
        .collect()
}

/// Display summary of discovered rocket components
#[allow(dead_code)]
pub fn display_rocket_summary(components: &HashMap<ComponentId, DiscoveredComponent>) {
    let rocket_components = filter_rocket_components(components);

    if !rocket_components.is_empty() {
        println!("\n🚀 Rocket Components Summary:");
        println!(
            "  {} rocket-specific components available",
            rocket_components.len()
        );

        // Group by category
        let mut categories: HashMap<&str, Vec<&DiscoveredComponent>> = HashMap::new();

        for comp in &rocket_components {
            let category = if comp.name.contains("world_") {
                "Position/Motion"
            } else if comp.name.contains("aero")
                || comp.name.contains("mach")
                || comp.name.contains("pressure")
            {
                "Aerodynamics"
            } else if comp.name.contains("thrust") || comp.name.contains("motor") {
                "Propulsion"
            } else if comp.name.contains("fin") || comp.name.contains("control") {
                "Control"
            } else {
                "Other"
            };

            categories.entry(category).or_default().push(comp);
        }

        for (category, comps) in categories {
            println!("\n  {}:", category.bold());
            for comp in comps {
                println!("    • {}", comp.name);
            }
        }
    } else {
        println!("\n⚠️  No rocket components found. Make sure rocket.py is running!");
    }
}
