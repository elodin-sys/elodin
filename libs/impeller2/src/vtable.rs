//! VTables allow you to dynamically formulate and parse tables.
//!
//! A VTable is made up of a series of "fields", each with an associated offset and length.
//! Each field has an associated expression that describes the associated component and entity id.
//! Expressions can also contain metadata for the field, like the schema, location of the timestamp,
//! or even the rate it should be populated at.
//!
//! VTables are used by `elodin-db` to let the user send data and receive data in the format and shape
//! they expect. This let's you specify a struct in Rust and send it directly into the db, with no serialization step.
//! `elodin-db` also allows you to subscribe to a particular VTable. The database will handle building tables from each of the fields
//! described in the VTable.
//!
//! At this point it is likely easier to see an example of how to formulate a VTable.
//! ```rust
//! use impeller2::types::PrimType;
//! use impeller2::vtable::VTable;
//! use impeller2::vtable::builder::*;
//! struct IMU {
//!     timestamp: u64,
//!     gyro: [f64; 3],
//!     accel: [f64; 3],
//! }
//!
//! impl IMU {
//!     fn as_vtable() -> VTable {
//!         let time = table!(IMU::timestamp);
//!         vtable([
//!             field!(
//!                 IMU::gyro,
//!                 schema(
//!                     PrimType::F64,
//!                     &[3],
//!                     timestamp(time.clone(), component("gyro")),
//!                 ),
//!             ),
//!             field!(
//!                 IMU::accel,
//!                 schema(
//!                     PrimType::F64,
//!                     &[3],
//!                     timestamp(time.clone(), component("accel")),
//!                 ),
//!             ),
//!         ])
//!     }
//! }
//! ```

use core::ops::Range;

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};
use zerocopy::{FromBytes, IntoBytes, TryFromBytes};

use crate::{
    buf::Buf,
    com_de::Decomponentize,
    error::Error,
    types::{ComponentId, ComponentView, PacketId, PrimType, Timestamp},
};

/// Well-known extension ID for nanosecond timestamp sources.
///
/// When the DB encounters an `OpExt` with this ID, it reads the source data as
/// nanoseconds and divides by 1000 to produce the microsecond record timestamp.
/// Use the [`builder::timestamp_ns`] convenience function to create this operation.
///
/// **Reserved ID:** `[0x01, 0x00]` is outside the auto-generated FNV1a hash range
/// and the manually-assigned `[224, *]` range used by other messages and extensions.
/// Do not reuse this ID for other extensions.
pub const TIMESTAMP_NS_EXT_ID: PacketId = [0x01, 0x00];

/// Operations that can be performed in a VTable
///
/// Each operation represents a different way to reference or manipulate data within a VTable.
#[derive(Debug, Serialize, Deserialize, Clone, postcard_schema::Schema)]
#[repr(u8)]
pub enum Op {
    Data {
        offset: Offset,
        len: u16,
    },
    Table {
        offset: Offset,
        len: u16,
    },
    None,
    Component {
        component_id: OpRef,
    },
    Schema {
        ty: OpRef,
        dim: OpRef,
        arg: OpRef,
    },

    Timestamp {
        source: OpRef,
        arg: OpRef,
    },

    Ext {
        arg: OpRef,
        id: PacketId,
        data: OpRef,
    },
}

/// A field within a VTable
///
/// Each field has an offset, length, and an associated operation reference.
#[derive(Debug, Serialize, Deserialize, Clone, postcard_schema::Schema)]
pub struct Field {
    pub offset: Offset,
    pub len: u16,
    pub arg: OpRef,
}

const _ASSERT_OP_SIZE: () = const {
    assert!(core::mem::size_of::<Op>() <= 8);
};

/// A reference to an operation in the VTable's operation list
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, postcard_schema::Schema,
)]
#[repr(transparent)]
pub struct OpRef(u16);

impl OpRef {
    /// Converts the operation reference to a usable index
    fn to_index(self) -> usize {
        self.0 as usize
    }
}

/// Represents an offset position within a buffer
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, postcard_schema::Schema,
)]
#[repr(transparent)]
pub struct Offset(u16);

impl From<u16> for Offset {
    fn from(val: u16) -> Self {
        Offset(val)
    }
}

impl Offset {
    /// Converts the offset to a usable index
    pub fn to_index(self) -> usize {
        self.0 as usize
    }
}

#[cfg(feature = "alloc")]
type DefaultOps = alloc::vec::Vec<Op>;

#[cfg(not(feature = "alloc"))]
type DefaultOps = heapless::Vec<Op, 32>;

#[cfg(feature = "alloc")]
type DefaultData = alloc::vec::Vec<u8>;
#[cfg(not(feature = "alloc"))]
type DefaultData = heapless::Vec<u8, 32>;
#[cfg(feature = "alloc")]
type DefaultFields = alloc::vec::Vec<Field>;
#[cfg(not(feature = "alloc"))]
type DefaultFields = heapless::Vec<Field, 32>;

/// A description of the layout of a table
#[derive(Debug, Serialize, Deserialize, Clone, Default, postcard_schema::Schema)]
pub struct VTable<
    Ops: Buf<Op> = DefaultOps,
    Data: Buf<u8> = DefaultData,
    Fields: Buf<Field> = DefaultFields,
> {
    #[serde(bound(deserialize = ""))]
    pub ops: Ops,
    #[serde(bound(deserialize = ""))]
    pub fields: Fields,
    #[serde(bound(deserialize = ""))]
    pub data: Data,
}

/// A component ID realized from an operation
#[derive(Clone, Copy, Debug)]
pub struct RealizedComponent {
    pub component_id: ComponentId,
}

/// A schema realized from an operation, containing dimension, type, and argument information
pub struct RealizedSchema<'a> {
    pub dim: &'a [u64],
    pub ty: PrimType,
    pub arg: OpRef,
}

/// A timestamp realized from an operation, possibly including range information
pub struct RealizedTimestamp {
    pub timestamp: Option<Timestamp>,
    pub range: Option<Range<usize>>,
    pub arg: OpRef,
    /// True when the source is nanoseconds (timestamp_ns ext); divide by 1000 for microseconds.
    pub is_ns: bool,
}

/// A slice of a table realized from an operation
pub struct RealizedTableSlice<'a> {
    pub slice: Option<&'a [u8]>,
    pub range: Range<usize>,
}

/// An extension realized from an operation, containing ID, data, and argument information
pub struct RealizedExt<'a> {
    pub id: PacketId,
    pub data: &'a [u8],
    /// The table byte range of the data source, if available.
    /// Present when the ext's data op is a `Table` reference (e.g. for `timestamp_ns`).
    pub range: Option<Range<usize>>,
    pub arg: OpRef,
}

/// A realized operation, which is an operation that has been evaluated against a table
pub enum RealizedOp<'a> {
    Data(&'a [u8]),
    Table(RealizedTableSlice<'a>),
    Component(RealizedComponent),
    Schema(RealizedSchema<'a>),
    Timestamp(RealizedTimestamp),
    Ext(RealizedExt<'a>),
    None,
}

/// A field realized from a VTable, containing entity and component IDs,
/// shape, type, component view, and timestamp information
pub struct RealizedField<'a> {
    pub component_id: ComponentId,
    pub shape: &'a [usize],
    /// Schema dimensions backing [`Self::shape`].
    pub dim: &'a [u64],
    pub ty: PrimType,
    pub view: Option<ComponentView<'a>>,
    pub timestamp: Option<Timestamp>,
    /// When set, the row timestamp must be re-read from this byte range on each table packet.
    pub timestamp_source_range: Option<Range<usize>>,
    /// When `true`, [`Self::timestamp_source_range`] holds nanoseconds (`/ 1000` for microseconds).
    pub timestamp_source_is_ns: bool,
}

/// Precomputed per-field dispatch for [`apply_table_dispatch_plan`].
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct TableDispatchEntry {
    pub component_id: ComponentId,
    pub ty: PrimType,
    pub dim: Vec<u64>,
    pub value_offset: usize,
    pub value_len: usize,
    pub timestamp_fixed: Option<Timestamp>,
    pub timestamp_source_range: Option<Range<usize>>,
    pub timestamp_source_is_ns: bool,
}

impl<'a> RealizedOp<'a> {
    /// Returns the operation as a byte slice if it contains data
    pub fn as_slice(&self) -> Option<&'a [u8]> {
        match self {
            RealizedOp::Data(data) => Some(data),
            RealizedOp::Table(table) => table.slice,
            _ => None,
        }
    }

    /// Attempts to interpret the operation as a component ID
    pub fn as_component_id(&self) -> Option<ComponentId> {
        let data = self.as_slice()?;
        let data: [u8; 8] = data.try_into().ok()?;
        Some(ComponentId(u64::from_le_bytes(data)))
    }

    /// Attempts to interpret the operation as a primitive type
    pub fn as_prim_ty(&self) -> Option<PrimType> {
        let data = self.as_slice()?;
        PrimType::try_read_from_bytes(data).ok()
    }

    /// Returns a reference to the realized timestamp if this operation is a timestamp
    pub fn as_timestamp(&self) -> Option<&RealizedTimestamp> {
        match self {
            RealizedOp::Timestamp(timestamp) => Some(timestamp),
            _ => None,
        }
    }

    /// Returns the range of a table if this operation represents a table
    pub fn as_table_range(&self) -> Option<Range<usize>> {
        match self {
            RealizedOp::Table(t) => Some(t.range.clone()),
            _ => None,
        }
    }
}

impl<Ops: Buf<Op>, Data: Buf<u8>, Fields: Buf<Field>> VTable<Ops, Data, Fields> {
    /// Retrieves an operation by its reference
    #[inline]
    pub fn get_op(&self, op_ref: OpRef) -> Result<&Op, Error> {
        self.ops
            .as_slice()
            .get(op_ref.to_index())
            .ok_or(Error::OpRefNotFound)
    }

    /// Evaluates an operation, and turns it into a [`RealizedOp`]
    ///
    /// `realize` follows each [`Offset`] and transforms it into an actual reference or data.
    /// Evaluates an operation, turning it into a [`RealizedOp`]
    ///
    /// `realize` follows each [`Offset`] and transforms it into an actual reference or data
    pub fn realize<'a>(
        &'a self,
        op_ref: OpRef,
        table: Option<&'a [u8]>,
    ) -> Result<RealizedOp<'a>, Error> {
        let op = self.get_op(op_ref)?;
        match op {
            Op::Data { offset, len } => {
                let data = self.data.as_slice();
                let data = data
                    .get(offset.to_index()..offset.to_index() + *len as usize)
                    .ok_or(Error::BufferOverflow)?;
                Ok(RealizedOp::Data(data))
            }
            Op::Table { offset, len } => {
                let range = offset.to_index()..offset.to_index() + *len as usize;
                let table = if let Some(table) = table {
                    Some(table.get(range.clone()).ok_or(Error::BufferUnderflow)?)
                } else {
                    None
                };
                Ok(RealizedOp::Table(RealizedTableSlice {
                    slice: table,
                    range,
                }))
            }
            Op::Component { component_id } => {
                let component_id = self
                    .realize(*component_id, table)?
                    .as_component_id()
                    .ok_or(Error::InvalidOp)?;
                Ok(RealizedOp::Component(RealizedComponent { component_id }))
            }
            Op::Schema { ty, dim, arg } => {
                let ty = self
                    .realize(*ty, table)?
                    .as_prim_ty()
                    .ok_or(Error::InvalidOp)?;
                let dim = self
                    .realize(*dim, table)?
                    .as_slice()
                    .ok_or(Error::InvalidOp)?;
                let dim = <[u64]>::try_ref_from_bytes(dim)?;
                Ok(RealizedOp::Schema(RealizedSchema { ty, dim, arg: *arg }))
            }
            Op::Timestamp { source, arg } => {
                let source = self.realize(*source, table)?;
                let timestamp = if let Some(data) = source.as_slice() {
                    Some(Timestamp::read_from_bytes(data)?)
                } else {
                    None
                };
                Ok(RealizedOp::Timestamp(RealizedTimestamp {
                    timestamp,
                    arg: *arg,
                    range: source.as_table_range(),
                    is_ns: false,
                }))
            }
            Op::None => Ok(RealizedOp::None),
            Op::Ext { arg, id, data } => {
                let resolved = self.realize(*data, table)?;
                let range = resolved.as_table_range();
                // When table is None (e.g. during VTable registration or
                // streaming setup), table-reference data operands can't
                // resolve to bytes. This is expected for timestamp_ns which
                // uses a raw_table() as its data source. Other exts
                // (MeanOp, FixedRateBehavior) use embedded Data ops that
                // always resolve, so this fallback only activates for
                // table-reference data.
                let data_bytes = resolved.as_slice().unwrap_or(&[]);
                Ok(RealizedOp::Ext(RealizedExt {
                    id: *id,
                    data: data_bytes,
                    range,
                    arg: *arg,
                }))
            }
        }
    }

    /// Evaluates a single [`Field`] the same way [`Self::realize_fields`] does.
    pub fn realize_field<'a>(
        &'a self,
        field: &'a Field,
        table: Option<&'a [u8]>,
    ) -> Result<RealizedField<'a>, Error> {
        let mut realized_op = self.realize(field.arg, table)?;
        let mut timestamp: Option<RealizedTimestamp> = None;
        let mut schema: Option<RealizedSchema<'_>> = None;
        loop {
            match realized_op {
                RealizedOp::Component(ref component) => {
                    let RealizedComponent { component_id } = *component;

                    let schema = schema.as_ref().ok_or(Error::SchemaNotFound)?;
                    // NOTE(sphw): bogan version of zerocopy::transmute_ref
                    // In the future this will need to also support 32 bit systems
                    // remove when https://github.com/google/zerocopy/pull/2428 is merged and released
                    let shape: &[usize] = <[usize]>::ref_from_bytes(schema.dim.as_bytes())?;
                    let view = if let Some(table) = table {
                        let offset = field.offset.to_index();
                        let data = table
                            .get(offset..offset + field.len as usize)
                            .ok_or(Error::BufferUnderflow)?;
                        Some(ComponentView::try_from_bytes_shape(data, shape, schema.ty)?)
                    } else {
                        None
                    };
                    let (timestamp, timestamp_source_range, timestamp_source_is_ns) =
                        match &timestamp {
                            None => (None, None, false),
                            Some(t) => {
                                if t.range.is_some() {
                                    (None, t.range.clone(), t.is_ns)
                                } else {
                                    (t.timestamp, None, false)
                                }
                            }
                        };
                    return Ok(RealizedField {
                        component_id,
                        view,
                        timestamp,
                        timestamp_source_range,
                        timestamp_source_is_ns,
                        shape,
                        dim: schema.dim,
                        ty: schema.ty,
                    });
                }
                RealizedOp::Schema(s) => {
                    let s = schema.insert(s);
                    realized_op = self.realize(s.arg, table)?;
                }
                RealizedOp::Timestamp(t) => {
                    let t = timestamp.insert(t);
                    realized_op = self.realize(t.arg, table)?;
                }
                RealizedOp::Ext(e) => {
                    if e.id == TIMESTAMP_NS_EXT_ID {
                        // Convert nanosecond timestamp to microseconds.
                        // When table is None (e.g. during VTable registration),
                        // e.data is empty so we skip reading the value -- only
                        // the structural walk matters in that case.
                        let ts = if e.data.len() >= core::mem::size_of::<Timestamp>() {
                            let ns = Timestamp::read_from_bytes(e.data)?;
                            Some(Timestamp(ns.0 / 1000))
                        } else {
                            None
                        };
                        timestamp = Some(RealizedTimestamp {
                            timestamp: ts,
                            arg: e.arg,
                            range: e.range.clone(),
                            is_ns: true,
                        });
                    }
                    realized_op = self.realize(e.arg, table)?;
                }
                _ => return Err(Error::InvalidOp),
            }
        }
    }

    /// Evaluated each `field`, returning a `RealizedField`
    ///
    /// `realized_fields` loops through each field, turning each [`Offset`] into a reference, and evaluating any [`Op`]
    /// Evaluates each field in the VTable, returning an iterator of [`RealizedField`]s
    ///
    /// This turns each [`Offset`] into a reference and evaluates any [`Op`]s
    pub fn realize_fields<'a>(
        &'a self,
        table: Option<&'a [u8]>,
    ) -> impl Iterator<Item = Result<RealizedField<'a>, Error>> + 'a {
        self.fields
            .iter()
            .map(move |field| self.realize_field(field, table))
    }

    /// Parses the passed in table, and sinks the values into the sink
    /// Parses the provided table and applies the values to the sink
    ///
    /// This evaluates each field in the VTable against the provided table data and
    /// applies the resulting values to the sink
    pub fn apply<D: Decomponentize>(
        &self,
        table: &[u8],
        sink: &mut D,
    ) -> Result<Result<(), D::Error>, Error> {
        for res in self.realize_fields(Some(table)) {
            let rf = res?;
            let ts = Self::resolve_apply_timestamp(table, &rf)?;
            let view = rf.view.expect("table not found");
            if let Err(err) = sink.apply_value(rf.component_id, view, ts) {
                return Ok(Err(err));
            }
        }
        Ok(Ok(()))
    }

    fn resolve_apply_timestamp(
        table: &[u8],
        rf: &RealizedField<'_>,
    ) -> Result<Option<Timestamp>, Error> {
        if let Some(r) = &rf.timestamp_source_range {
            let data = table
                .get(r.clone())
                .ok_or(Error::BufferUnderflow)?
                .get(..core::mem::size_of::<Timestamp>())
                .ok_or(Error::BufferUnderflow)?;
            let t = Timestamp::read_from_bytes(data)?;
            Ok(Some(if rf.timestamp_source_is_ns {
                Timestamp(t.0 / 1000)
            } else {
                t
            }))
        } else {
            Ok(rf.timestamp)
        }
    }
}

/// Applies a precomputed [`TableDispatchEntry`] plan (see [`VTable::build_table_dispatch_plan`]).
#[cfg(feature = "alloc")]
pub fn apply_table_dispatch_plan<D: Decomponentize>(
    plan: &[TableDispatchEntry],
    table: &[u8],
    sink: &mut D,
) -> Result<Result<(), D::Error>, Error> {
    for e in plan {
        let dim = e.dim.as_slice();
        let shape = <[usize]>::ref_from_bytes(dim.as_bytes())?;
        let data = table
            .get(e.value_offset..e.value_offset + e.value_len)
            .ok_or(Error::BufferUnderflow)?;
        let view = ComponentView::try_from_bytes_shape(data, shape, e.ty)?;
        let ts = if let Some(r) = &e.timestamp_source_range {
            let slice = table
                .get(r.clone())
                .ok_or(Error::BufferUnderflow)?
                .get(..core::mem::size_of::<Timestamp>())
                .ok_or(Error::BufferUnderflow)?;
            let t = Timestamp::read_from_bytes(slice)?;
            Some(if e.timestamp_source_is_ns {
                Timestamp(t.0 / 1000)
            } else {
                t
            })
        } else {
            e.timestamp_fixed
        };
        if let Err(err) = sink.apply_value(e.component_id, view, ts) {
            return Ok(Err(err));
        }
    }
    Ok(Ok(()))
}

#[cfg(feature = "alloc")]
impl<Ops: Buf<Op>, Data: Buf<u8>, Fields: Buf<Field>> VTable<Ops, Data, Fields> {
    /// Builds a static dispatch plan for repeated `apply` on table buffers of sufficient length.
    pub fn build_table_dispatch_plan(&self) -> Result<Vec<TableDispatchEntry>, Error> {
        let mut need = 0usize;
        for field in self.fields.iter() {
            need = need.max(field.offset.to_index() + field.len as usize);
        }
        for op in self.ops.as_slice() {
            if let Op::Table { offset, len } = op {
                need = need.max(offset.to_index() + *len as usize);
            }
        }
        let mut probe = vec![0u8; need.max(1)];
        for _ in 0..64 {
            let mut out = Vec::with_capacity(self.fields.len());
            let mut ok = true;
            for field in self.fields.iter() {
                match self.realize_field(field, Some(&probe)) {
                    Ok(rf) => {
                        if rf.view.is_none() {
                            ok = false;
                            break;
                        }
                        let offset = field.offset.to_index();
                        out.push(TableDispatchEntry {
                            component_id: rf.component_id,
                            ty: rf.ty,
                            dim: rf.dim.to_vec(),
                            value_offset: offset,
                            value_len: field.len as usize,
                            timestamp_fixed: rf.timestamp,
                            timestamp_source_range: rf.timestamp_source_range.clone(),
                            timestamp_source_is_ns: rf.timestamp_source_is_ns,
                        });
                    }
                    Err(Error::BufferUnderflow) => {
                        ok = false;
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
            if ok {
                return Ok(out);
            }
            let next = probe.len().saturating_mul(2).max(probe.len() + 256);
            probe.resize(next, 0);
        }
        Err(Error::InvalidOp)
    }
}

#[cfg(feature = "alloc")]
/// Tools for building VTables programmatically
pub mod builder {
    use alloc::collections::BTreeMap;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use zerocopy::Immutable;

    use super::*;
    /// A builder for VTable operations
    pub enum OpBuilder {
        Data {
            align: usize,
            data: Vec<u8>,
        },
        Table {
            offset: Offset,
            len: u16,
        },
        Component {
            component_id: Arc<OpBuilder>,
        },
        Schema {
            ty: Arc<OpBuilder>,
            dim: Arc<OpBuilder>,
            arg: Arc<OpBuilder>,
        },
        Timestamp {
            timestamp: Arc<OpBuilder>,
            arg: Arc<OpBuilder>,
        },
        Ext {
            id: PacketId,
            data: Arc<OpBuilder>,
            arg: Arc<OpBuilder>,
        },
    }

    /// A builder for VTable fields
    #[derive(Clone)]
    pub struct FieldBuilder {
        offset: Offset,
        len: u16,
        arg: Arc<OpBuilder>,
    }

    /// Creates a data operation builder from the provided data
    pub fn data<T: IntoBytes + Immutable + ?Sized>(data: &T) -> Arc<OpBuilder> {
        let align = core::mem::align_of_val(data);
        Arc::new(OpBuilder::Data {
            align,
            data: data.as_bytes().to_vec(),
        })
    }

    /// Creates a table operation builder with the specified offset and length
    pub fn raw_table(offset: impl Into<Offset>, len: u16) -> Arc<OpBuilder> {
        Arc::new(OpBuilder::Table {
            offset: offset.into(),
            len,
        })
    }

    /// Creates a component operation builder from a component ID
    pub fn component(component_id: impl Into<ComponentId>) -> Arc<OpBuilder> {
        let component_id = component_id.into();
        let component_id = data(&component_id);
        Arc::new(OpBuilder::Component { component_id })
    }

    /// Creates a schema operation builder from a primitive type, dimensions, and an argument
    pub fn schema(ty: PrimType, dim: &[u64], arg: Arc<OpBuilder>) -> Arc<OpBuilder> {
        let ty = data(&ty);
        let dim = data(dim);
        Arc::new(OpBuilder::Schema { ty, dim, arg })
    }

    /// Creates a timestamp operation builder from a timestamp source and an argument
    pub fn timestamp(timestamp: Arc<OpBuilder>, arg: Arc<OpBuilder>) -> Arc<OpBuilder> {
        Arc::new(OpBuilder::Timestamp { timestamp, arg })
    }

    /// Creates a nanosecond-source timestamp operation.
    ///
    /// The DB engine divides the source value by 1000 to produce microseconds
    /// for the record timestamp. The raw component data is stored unchanged.
    /// This uses the existing `OpExt` wire format with the well-known
    /// [`TIMESTAMP_NS_EXT_ID`] extension ID.
    pub fn timestamp_ns(source: Arc<OpBuilder>, arg: Arc<OpBuilder>) -> Arc<OpBuilder> {
        Arc::new(OpBuilder::Ext {
            id: TIMESTAMP_NS_EXT_ID,
            data: source,
            arg,
        })
    }

    /// Creates an extension operation builder from a message and an argument
    ///
    ///  Extensions are used to attache extra metadata to a field.
    ///  For instance elodin-db uses extensions to specify the rate you want to receive a field at.
    pub fn ext<D: crate::types::Msg>(data: D, arg: Arc<OpBuilder>) -> Arc<OpBuilder> {
        let data = postcard::to_allocvec(&data).expect("data serialize failed");
        let data = Arc::new(OpBuilder::Data { align: 1, data });
        Arc::new(OpBuilder::Ext {
            id: D::ID,
            data,
            arg,
        })
    }

    /// Creates a field builder with the specified offset, length, and argument
    pub fn raw_field(offset: impl Into<Offset>, len: u16, arg: Arc<OpBuilder>) -> FieldBuilder {
        FieldBuilder {
            offset: offset.into(),
            len,
            arg,
        }
    }

    #[macro_export]
    /// Creates a `Op::Table` from the specified struct field
    macro_rules! table {
        ($t:tt::$field:ident $(,)?) => {{
            let offset = core::mem::offset_of!($t, $field);
            let size = $crate::vtable::builder::field_size!($t, $field) as u16;
            raw_table(offset as u16, size)
        }};
    }

    pub use table;

    /// Creates a [`Field`] from the specified struct field, and ops
    ///
    /// # Usage
    /// ```rust
    /// use impeller2::vtable::builder::*;
    /// struct Foo { bar: [f64; 3] }
    /// vtable([field!(Foo::bar, component("bar"))]);
    /// ```
    #[macro_export]
    macro_rules! field {
        ($t:tt::$field:ident, $arg: expr $(,)?) => {{
            let offset = core::mem::offset_of!($t, $field);
            let size = $crate::vtable::builder::field_size!($t, $field) as u16;
            $crate::vtable::builder::raw_field(offset as u16, size, $arg)
        }};
    }

    pub use field;

    /// Returns the size of a field of a struct
    ///
    /// # Usage
    /// ```rust
    /// struct Foo { bar: [f64; 3]}
    /// assert_eq!(impeller2::vtable::builder::field_size!(Foo, bar), 3 * 8)
    /// ```
    ///
    /// source: <https://stackoverflow.com/a/70222282>
    #[macro_export]
    macro_rules! field_size {
        ($t:ident, $field:ident) => {{
            let m = core::mem::MaybeUninit::<$t>::uninit();
            // According to https://doc.rust-lang.org/stable/std/ptr/macro.addr_of_mut.html#examples,
            // you can dereference an uninitialized MaybeUninit pointer in addr_of!
            // Raw pointer deref in const contexts is stabilized in 1.58:
            // https://github.com/rust-lang/rust/pull/89551
            let p = unsafe {
                core::ptr::addr_of!((*(&m as *const _ as *const $t)).$field)
            };

            const fn size_of_raw<T>(_: *const T) -> usize {
                core::mem::size_of::<T>()
            }
            size_of_raw(p)
        }};
    }

    pub use field_size;

    /// A builder for constructing VTables.
    ///
    /// For most cases you should use [`vtable`] instead
    #[derive(Default)]
    pub struct VTableBuilder {
        vtable: VTable<Vec<Op>, Vec<u8>, Vec<Field>>,
        visited: BTreeMap<usize, OpRef>,
    }

    impl VTableBuilder {
        /// Visits an operation builder, adding it to the VTable and returning its `OpRef`
        pub fn visit(&mut self, op: &Arc<OpBuilder>) -> OpRef {
            let id = Arc::as_ptr(op) as usize;
            if let Some(r) = self.visited.get(&id) {
                return *r;
            }
            let op = match op.as_ref() {
                OpBuilder::Data { align, data } => {
                    let len = data.len() as u16;
                    let padding = (align - (self.vtable.data.len() % align)) % align;

                    for _ in 0..padding {
                        self.vtable.data.push(0);
                    }
                    let offset = Offset(self.vtable.data.len() as u16);
                    self.vtable.data.extend_from_slice(data);
                    Op::Data { offset, len }
                }
                OpBuilder::Table { offset, len } => Op::Table {
                    offset: *offset,
                    len: *len,
                },
                OpBuilder::Component { component_id } => {
                    let component_id = self.visit(component_id);
                    Op::Component { component_id }
                }
                OpBuilder::Schema { ty, dim, arg } => {
                    let ty = self.visit(ty);
                    let dim = self.visit(dim);
                    let arg = self.visit(arg);
                    Op::Schema { ty, dim, arg }
                }
                OpBuilder::Timestamp { timestamp, arg } => {
                    let source = self.visit(timestamp);
                    let arg = self.visit(arg);
                    Op::Timestamp { source, arg }
                }
                OpBuilder::Ext { id, data, arg } => {
                    let arg = self.visit(arg);
                    let data = self.visit(data);
                    Op::Ext { id: *id, data, arg }
                }
            };
            let op_ref = OpRef(self.vtable.ops.len() as u16);
            self.vtable.ops.push(op);
            self.visited.insert(id, op_ref);
            op_ref
        }
    }

    /// Creates a VTable from the provided field builders
    pub fn vtable(
        fields: impl IntoIterator<Item = FieldBuilder>,
    ) -> VTable<Vec<Op>, Vec<u8>, Vec<Field>> {
        let mut builder = VTableBuilder::default();
        for field in fields.into_iter() {
            let field = Field {
                offset: field.offset,
                len: field.len,
                arg: builder.visit(&field.arg),
            };
            builder.vtable.fields.push(field);
        }
        builder.vtable
    }
}

#[cfg(feature = "alloc")]
pub trait AsVTable {
    fn populate_vtable_fields(
        builder: &mut alloc::vec::Vec<builder::FieldBuilder>,
    ) -> Result<(), crate::error::Error>;

    fn as_vtable() -> VTable {
        let mut fields = alloc::vec::Vec::new();
        Self::populate_vtable_fields(&mut fields).expect("vtable failed to form");
        builder::vtable(fields)
    }
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;
    use nox::array::ArrayViewExt;
    use std::collections::HashMap;

    use nox::{Array, ArrayBuf, Dyn};
    use zerocopy::{Immutable, IntoBytes};

    use crate::{
        com_de::Decomponentize,
        types::{ComponentId, ComponentView, PrimType, Timestamp},
    };

    #[derive(Default)]
    struct TestSink {
        timestamp: Option<Timestamp>,
        f32_components: HashMap<ComponentId, Array<f32, Dyn>>,
        f64_components: HashMap<ComponentId, Array<f64, Dyn>>,
    }

    impl Decomponentize for TestSink {
        type Error = Infallible;
        fn apply_value(
            &mut self,
            component_id: ComponentId,
            value: ComponentView<'_>,
            timestamp: Option<Timestamp>,
        ) -> Result<(), Self::Error> {
            if let Some(timestamp) = timestamp {
                self.timestamp = Some(timestamp);
            }
            match value {
                ComponentView::F32(view) => {
                    self.f32_components
                        .insert(component_id, view.to_dyn_owned());
                }

                ComponentView::F64(view) => {
                    self.f64_components
                        .insert(component_id, view.to_dyn_owned());
                }
                _ => todo!(),
            }
            Ok(())
        }
    }

    #[test]
    fn test_basic_builder() {
        use super::builder::*;

        #[derive(IntoBytes, Immutable)]
        struct Foo {
            timestamp: Timestamp,
            test: [f32; 4],
            bar: f64,
        }

        let time = table!(Foo::timestamp);
        let v = vtable([
            field!(
                Foo::test,
                schema(
                    PrimType::F32,
                    &[4],
                    timestamp(time.clone(), component("test")),
                ),
            ),
            field!(
                Foo::bar,
                schema(PrimType::F64, &[], timestamp(time, component("bar")))
            ),
        ]);

        let foo = Foo {
            timestamp: Timestamp(1000),
            test: [1.0, 2.0, 3.0, 4.0],
            bar: 5.0,
        };
        let mut sink = TestSink::default();
        v.apply(foo.as_bytes(), &mut sink).unwrap().unwrap();
        let test = sink.f32_components.get(&ComponentId::new("test")).unwrap();
        assert_eq!(test.buf.as_buf(), &[1.0, 2.0, 3.0, 4.0]);

        let bar = sink.f64_components.get(&ComponentId::new("bar")).unwrap();
        assert_eq!(bar.buf.as_buf(), &[5.0]);
        assert_eq!(sink.timestamp, Some(foo.timestamp));
    }

    #[test]
    fn test_dispatch_plan_matches_apply() {
        use super::apply_table_dispatch_plan;
        use super::builder::*;

        #[derive(IntoBytes, Immutable)]
        struct Foo {
            timestamp: Timestamp,
            test: [f32; 4],
            bar: f64,
        }

        let time = table!(Foo::timestamp);
        let v = vtable([
            field!(
                Foo::test,
                schema(
                    PrimType::F32,
                    &[4],
                    timestamp(time.clone(), component("test")),
                ),
            ),
            field!(
                Foo::bar,
                schema(PrimType::F64, &[], timestamp(time, component("bar")))
            ),
        ]);

        let plan = v.build_table_dispatch_plan().expect("plan");
        let foo = Foo {
            timestamp: Timestamp(1000),
            test: [1.0, 2.0, 3.0, 4.0],
            bar: 5.0,
        };
        let bytes = foo.as_bytes();

        let mut sink_apply = TestSink::default();
        v.apply(bytes, &mut sink_apply).unwrap().unwrap();
        let mut sink_plan = TestSink::default();
        apply_table_dispatch_plan(&plan, bytes, &mut sink_plan)
            .unwrap()
            .unwrap();

        assert_eq!(sink_apply.timestamp, sink_plan.timestamp);
        assert_eq!(
            sink_apply.f32_components.get(&ComponentId::new("test")),
            sink_plan.f32_components.get(&ComponentId::new("test"))
        );
        assert_eq!(
            sink_apply.f64_components.get(&ComponentId::new("bar")),
            sink_plan.f64_components.get(&ComponentId::new("bar"))
        );
    }
}
