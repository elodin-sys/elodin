//! The catalog of an export: its tables, their columns, and every exported signal.
//!
//! Written by `elodin-db export --manifest` and built for local recordings by the MCP host,
//! so local and Elodin Cloud searches see the same shapes. Private components never appear.

use std::collections::BTreeMap;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Int64Array, RecordBatch, TimestampMicrosecondArray,
};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema};
use impeller::types::PrimType;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::export::{self, ExportPlan};
use crate::{DB, Error};

pub const VERSION: u32 = 1;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Catalog {
    pub version: u32,
    pub join: bool,
    pub pattern: Option<String>,
    pub tables: Vec<Table>,
    pub signals: Vec<Signal>,
}

/// One export file. `rows == 0` means export skips it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Table {
    pub name: String,
    pub rows: u64,
    pub columns: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Signal {
    /// The component's original (not lowercased) name.
    pub name: String,
    pub table: String,
    pub column: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub elements: Vec<String>,
    /// Recorded samples; an outer-join table can hold fewer non-null rows.
    pub rows: u64,
    pub first_us: Option<i64>,
    pub last_us: Option<i64>,
    pub timestamp_source: bool,
    pub metadata: BTreeMap<String, String>,
}

impl Catalog {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("catalog serializes")
    }

    pub fn table(&self, name: &str) -> Option<&Table> {
        self.tables
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(name))
    }

    pub fn signal(&self, name: &str) -> Option<&Signal> {
        self.signals
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case(name))
    }
}

/// The catalog of `db` as `export --mono-us [--join] [--pattern]` would lay it out.
pub fn catalog(db: &DB, join: bool, pattern: Option<&str>) -> Result<Catalog, Error> {
    let plan = export::plan(db, join, pattern, false)?;
    Ok(from_plan(&plan, join, pattern.map(String::from)))
}

pub(crate) fn from_plan(plan: &ExportPlan, join: bool, pattern: Option<String>) -> Catalog {
    let mut tables = Vec::with_capacity(plan.groups.len());
    let mut signals = Vec::new();
    for group in &plan.groups {
        tables.push(Table {
            name: group.output_name.clone(),
            rows: export::group_batch(group).num_rows() as u64,
            columns: group.members.iter().map(|m| m.short_name.clone()).collect(),
        });
        for m in &group.members {
            let times = m.component.as_time_series_array();
            let rows = times.len().min(m.component.as_data_array("").1.len());
            let times = times
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("time array type")
                .slice(0, rows);
            signals.push(Signal {
                name: m.metadata.name.clone(),
                table: group.output_name.clone(),
                column: m.short_name.clone(),
                dtype: dtype_name(m.component.schema.prim_type).to_string(),
                shape: m.component.schema.dim.to_vec(),
                elements: m
                    .element_names
                    .split(',')
                    .map(|e| e.trim().to_string())
                    .filter(|e| !e.is_empty())
                    .collect(),
                rows: rows as u64,
                first_us: compute::min(&times),
                last_us: compute::max(&times),
                timestamp_source: m.metadata.is_timestamp_source(),
                metadata: m
                    .metadata
                    .metadata
                    .iter()
                    .filter(|(k, _)| !k.starts_with('_') && *k != "element_names")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            });
        }
    }
    tables.sort_by(|a, b| a.name.cmp(&b.name));
    signals.sort_by(|a, b| a.name.cmp(&b.name));
    Catalog {
        version: VERSION,
        join,
        pattern,
        tables,
        signals,
    }
}

fn dtype_name(prim: PrimType) -> &'static str {
    match prim {
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

/// A read of one export table, filtered the way Elodin Cloud filters its lake copy.
#[derive(Clone, Debug, Default)]
pub struct Read {
    pub table: String,
    /// Member columns to return; empty means all of them.
    pub columns: Vec<String>,
    pub start_us: Option<i64>,
    pub end_us: Option<i64>,
    /// Keep only rows where this column is not null.
    pub nonnull: Option<String>,
}

/// `seq, time_us, <columns>` for the rows of `read.table` in export order, with
/// `start_us <= time_us < end_us` and `nonnull` not null. `None` if the table is unknown.
pub fn read(db: &DB, join: bool, read: &Read) -> Result<Option<RecordBatch>, Error> {
    let plan = export::plan(db, join, None, false)?;
    let Some(group) = plan
        .groups
        .iter()
        .find(|g| g.output_name.eq_ignore_ascii_case(&read.table))
    else {
        return Ok(None);
    };
    let batch = export::group_batch(group);
    let n = batch.num_rows();

    let mut fields = vec![Arc::new(Field::new("seq", DataType::Int64, false))];
    let mut cols: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter_values(0..n as i64))];
    fields.push(batch.schema().field(0).clone().into());
    cols.push(batch.column(0).clone());
    for (i, f) in batch.schema().fields().iter().enumerate().skip(1) {
        if read.columns.is_empty()
            || read
                .columns
                .iter()
                .any(|c| c.eq_ignore_ascii_case(f.name()))
        {
            fields.push(f.clone());
            cols.push(batch.column(i).clone());
        }
    }

    let time = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("time_us is Int64");
    let nonnull = match &read.nonnull {
        Some(c) => match batch.schema().index_of(&c.to_lowercase()) {
            Ok(i) => Some(batch.column(i).clone()),
            Err(_) => return Ok(None),
        },
        None => None,
    };
    let keep: BooleanArray = (0..n)
        .map(|i| {
            let t = time.value(i);
            Some(
                read.start_us.is_none_or(|s| t >= s)
                    && read.end_us.is_none_or(|e| t < e)
                    && nonnull.as_ref().is_none_or(|a| a.is_valid(i)),
            )
        })
        .collect();
    let out = RecordBatch::try_new(Arc::new(Schema::new(fields)), cols)?;
    Ok(Some(compute::filter_record_batch(&out, &keep)?))
}

/// Number of rows [`read`] would return, without materializing them.
pub fn count(db: &DB, join: bool, read: &Read) -> Result<Option<u64>, Error> {
    // The filter is cheap next to the join, so share the path with `read`.
    Ok(self::read(
        db,
        join,
        &Read {
            columns: vec!["__none__".into()],
            ..read.clone()
        },
    )?
    .map(|b| b.num_rows() as u64))
}
