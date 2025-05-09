use arrow::{
    array::{
        ArrayRef, ArrowPrimitiveType, BooleanArray, FixedSizeListArray, PrimitiveArray,
        RecordBatch, TimestampMicrosecondArray,
    },
    buffer::{BooleanBuffer, Buffer, ScalarBuffer},
    datatypes::*,
};
use convert_case::{Case, Casing};
use datafusion::{
    catalog::streaming::StreamingTable, datasource::MemTable, execution::RecordBatchStream,
    parquet, physical_plan::streaming::PartitionStream, prelude::SessionContext,
};
use futures_lite::{Stream, StreamExt, pin};
use impeller2::types::{PrimType, Timestamp};
use impeller2_wkt::ArchiveFormat;
use std::{
    fs::File,
    ops::{Bound, RangeBounds},
    path::Path,
    pin::Pin,
    ptr::NonNull,
    sync::Arc,
    task::{Context, Poll},
};
use zerocopy::{Immutable, IntoBytes};

use crate::{Component, DB, Error, append_log::AppendLog};

impl<T: IntoBytes + Immutable> AppendLog<T> {
    pub fn as_arrow_buffer(&self, element_size: usize) -> Buffer {
        self.as_arrow_buffer_range(.., element_size)
    }

    pub fn as_arrow_buffer_range<R: RangeBounds<usize>>(
        &self,
        range: R,
        element_size: usize,
    ) -> Buffer {
        let data = self.data();
        let start = match range.start_bound() {
            Bound::Included(&n) => n * element_size,
            Bound::Excluded(&n) => (n + 1) * element_size,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => (n + 1) * element_size,
            Bound::Excluded(&n) => n * element_size,
            Bound::Unbounded => data.len(),
        };
        let start = start.min(data.len());
        let end = end.min(data.len());
        let len = end.saturating_sub(start);

        unsafe {
            let ptr = NonNull::new(data.as_ptr().add(start) as *mut _).expect("mmap null");
            Buffer::from_custom_allocation(ptr, len, self.raw_mmap().clone())
        }
    }
}

impl Component {
    pub fn as_data_array(&self, name: impl ToString) -> (FieldRef, ArrayRef) {
        self.as_data_array_range(name, ..)
    }

    pub fn as_data_array_range<R: RangeBounds<usize>>(
        &self,
        name: impl ToString,
        range: R,
    ) -> (FieldRef, ArrayRef) {
        let size = self.schema.dim.iter().product::<usize>() as i32;
        let buf = self.time_series.data();
        let element_size = self.time_series.element_size();
        let array = match self.schema.prim_type {
            PrimType::F64 => array_ref::<Float64Type, _>(buf, range, element_size),
            PrimType::F32 => array_ref::<Float32Type, _>(buf, range, element_size),
            PrimType::U64 => array_ref::<UInt64Type, _>(buf, range, element_size),
            PrimType::U32 => array_ref::<UInt32Type, _>(buf, range, element_size),
            PrimType::U16 => array_ref::<UInt16Type, _>(buf, range, element_size),
            PrimType::U8 => array_ref::<UInt8Type, _>(buf, range, element_size),
            PrimType::I64 => array_ref::<Int64Type, _>(buf, range, element_size),
            PrimType::I32 => array_ref::<Int32Type, _>(buf, range, element_size),
            PrimType::I16 => array_ref::<Int16Type, _>(buf, range, element_size),
            PrimType::I8 => array_ref::<Int8Type, _>(buf, range, element_size),
            PrimType::Bool => bool_ref(buf, range, element_size),
        };

        let inner_field = Arc::new(Field::new(
            name.to_string(),
            array.data_type().clone(),
            false,
        ));

        if self.schema.dim.is_empty() {
            return (inner_field, array);
        }

        let data_type = match self.schema.prim_type {
            PrimType::U8 => DataType::UInt8,
            PrimType::U16 => DataType::UInt16,
            PrimType::U32 => DataType::UInt32,
            PrimType::U64 => DataType::UInt64,
            PrimType::I8 => DataType::Int8,
            PrimType::I16 => DataType::Int16,
            PrimType::I32 => DataType::Int32,
            PrimType::I64 => DataType::Int64,
            PrimType::Bool => DataType::Boolean,
            PrimType::F32 => DataType::Float32,
            PrimType::F64 => DataType::Float64,
        };

        let field = Arc::new(
            Field::new_fixed_size_list(name.to_string(), inner_field.clone(), size, false)
                .with_extension_type(
                    arrow_schema::extension::FixedShapeTensor::try_new(
                        data_type,
                        self.schema.dim.iter().copied(),
                        None,
                        None,
                    )
                    .unwrap(),
                ),
        );
        let array = FixedSizeListArray::new(inner_field, size, array, None);
        let array = Arc::new(array);
        (field, array)
    }

    pub fn as_time_series_array(&self) -> ArrayRef {
        self.as_time_series_array_range(..)
    }

    pub fn as_time_series_array_range<R: RangeBounds<usize>>(&self, range: R) -> ArrayRef {
        let buf = self.time_series.index();
        let array =
            scalar_buffer::<TimestampMicrosecondType, _>(buf, range, size_of::<Timestamp>());
        let array = TimestampMicrosecondArray::new(array, None);
        Arc::new(array)
    }

    pub fn as_record_batch(&self, name: impl ToString) -> RecordBatch {
        self.as_record_batch_range(name, ..)
    }

    pub fn as_record_batch_range(
        &self,
        name: impl ToString,
        range: impl RangeBounds<usize> + Clone,
    ) -> RecordBatch {
        let name = name.to_string();
        let (data_field, data_array) = self.as_data_array_range(name.clone(), range.clone());
        let time_array = self.as_time_series_array_range(range);
        let len = data_array.len().min(time_array.len());
        let time_field = Arc::new(Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ));
        let fields = vec![data_field, time_field];
        let columns = vec![data_array.slice(0, len), time_array.slice(0, len)];

        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
            .expect("record batch params wrong")
    }

    pub fn as_stream_table(&self, name: impl ToString) -> StreamingTable {
        let stream = Arc::new(ComponentPartStream::new(name.to_string(), self.clone()));
        StreamingTable::try_new(stream.schema.clone(), vec![stream])
            .expect("streaming table create failed")
            .with_infinite_table(true)
    }
    pub fn as_mem_table(&self, name: impl ToString) -> MemTable {
        let record_batch = self.as_record_batch(name);
        let schema = record_batch.schema();
        let record_batches = vec![record_batch];
        MemTable::try_new(schema, vec![record_batches])
            .expect("mem table create failed")
            .with_sort_order(vec![vec![datafusion::logical_expr::SortExpr::new(
                datafusion::prelude::col("time"),
                true,
                false,
            )]])
    }
}

impl DB {
    pub fn as_session_context(&self) -> Result<SessionContext, datafusion::error::DataFusionError> {
        use datafusion::prelude::*;
        let config = SessionConfig::new().set_bool("datafusion.catalog.information_schema", true);
        let ctx = SessionContext::new_with_config(config);
        self.with_state(|state| {
            for component in state.components.values() {
                let entity_metadata = state.entity_metadata.get(&component.entity_id).unwrap();
                let component_metadata = state
                    .component_metadata
                    .get(&component.component_id)
                    .unwrap();
                let entity_name = entity_metadata.name.to_case(convert_case::Case::Snake);
                let component_name = component_metadata.name.to_case(convert_case::Case::Snake);
                let name = format!("{entity_name}_{component_name}");
                let mem_table = component.as_mem_table(&component_name);
                ctx.register_table(name, Arc::new(mem_table))?;

                let stream_name = format!("{entity_name}_{component_name}_stream");
                let stream_table = component.as_stream_table(component_name);
                ctx.register_table(stream_name, Arc::new(stream_table))?;
            }
            Ok::<_, datafusion::error::DataFusionError>(())
        })?;
        Ok(ctx)
    }
    pub async fn insert_views(
        &self,
        ctx: &mut SessionContext,
    ) -> Result<(), datafusion::error::DataFusionError> {
        let queries = self.with_state(|state| {
            let mut queries = vec![];
            for (entity_id, entity_metadata) in state.entity_metadata.iter() {
                let entity_name = entity_metadata.name.to_case(Case::Snake);
                let mut view_query_end = "".to_string();
                let mut selects = vec![];
                let mut first_table = None;
                for component in state.components.values().filter(|c| c.entity_id == *entity_id) {
                    let component_metadata = state.component_metadata.get(&component.component_id).unwrap();
                    let component_name = component_metadata.name.to_case(Case::Snake);
                    let table = format!("{entity_name}_{component_name}");
                    selects.push(format!("{table:?}.{component_name:?}"));
                    if let Some(first_table) = &first_table {
                        view_query_end = format!(
                            "{view_query_end} FULL OUTER JOIN {table:?} on {first_table:?}.time = {table:?}.time"
                        );
                    } else {
                        first_table = Some(table);
                    };
                }
                let Some(first_table) = first_table else {
                    continue;
                };
                if selects.is_empty() {
                    continue;
                }
                selects.push(format!("{first_table:?}.time"));
                let selects = selects.join(",");
                let view_query = format!(
                    "CREATE VIEW {entity_name:?} AS SELECT {selects} FROM {first_table:?} {view_query_end}"
                );
                queries.push(view_query);
            }
            queries
        });
        for view_query in queries {
            ctx.sql(&view_query).await?.collect().await?;
        }
        Ok(())
    }

    pub fn save_archive(&self, path: impl AsRef<Path>, format: ArchiveFormat) -> Result<(), Error> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        self.with_state(|state| {
            for component in state.components.values() {
                let Some(component_metadata) =
                    state.component_metadata.get(&component.component_id)
                else {
                    continue;
                };
                let Some(entity_metadata) = state.entity_metadata.get(&component.entity_id) else {
                    continue;
                };

                let column_name = format!("{}_{}", entity_metadata.name, component_metadata.name);
                let record_batch = component.as_record_batch(column_name.clone());
                let schema = record_batch.schema_ref();
                match format {
                    ArchiveFormat::ArrowIpc => {
                        let file_name = format!("{column_name}.arrow");
                        let file_path = path.join(file_name);
                        let mut file = File::create(file_path)?;
                        let mut writer =
                            arrow::ipc::writer::FileWriter::try_new(&mut file, schema)?;
                        writer.write(&record_batch)?;
                        writer.finish()?;
                    }
                    #[cfg(feature = "parquet")]
                    ArchiveFormat::Parquet => {
                        let file_name = format!("{column_name}.parquet");
                        let file_path = path.join(file_name);
                        let mut file = File::create(file_path)?;
                        let mut writer =
                            parquet::arrow::ArrowWriter::try_new(&mut file, schema.clone(), None)?;
                        writer.write(&record_batch)?;
                        writer.close()?;
                    }
                    ArchiveFormat::Csv => {
                        let file_name = format!("{column_name}.csv");
                        let file_path = path.join(file_name);
                        let mut file = File::create(file_path)?;
                        let mut writer = arrow::csv::Writer::new(&mut file);
                        writer.write(&record_batch)?;
                    }
                    #[allow(unreachable_patterns)]
                    _ => return Err(Error::UnsupportedArchiveFormat),
                }
            }
            Ok(())
        })
    }
}

fn bool_ref<T: IntoBytes + Immutable, R: RangeBounds<usize>>(
    buf: &AppendLog<T>,
    range: R,
    element_size: usize,
) -> ArrayRef {
    let buffer = buf.as_arrow_buffer_range(range, element_size);
    let len = buffer.len();
    let buf = BooleanBuffer::new(buffer, 0, len);
    Arc::new(BooleanArray::new(buf, None))
}

fn array_ref<P: ArrowPrimitiveType, T: IntoBytes + Immutable>(
    buf: &AppendLog<T>,
    range: impl RangeBounds<usize>,
    element_size: usize,
) -> ArrayRef {
    let scalar_buffer = scalar_buffer::<P, T>(buf, range, element_size);
    let array = PrimitiveArray::<P>::new(scalar_buffer, None);
    Arc::new(array)
}

fn scalar_buffer<P: ArrowPrimitiveType, T: IntoBytes + Immutable>(
    buf: &AppendLog<T>,
    range: impl RangeBounds<usize>,
    element_size: usize,
) -> ScalarBuffer<P::Native> {
    let buffer = buf.as_arrow_buffer_range(range, element_size);
    let len = buffer.len() / std::mem::size_of::<P::Native>();
    ScalarBuffer::<P::Native>::new(buffer, 0, len)
}

#[pin_project::pin_project]
pub struct ComponentStream {
    stream: Pin<
        Box<
            dyn Stream<Item = Result<RecordBatch, datafusion::error::DataFusionError>>
                + Send
                + Sync,
        >,
    >,
    schema: SchemaRef,
}

pub struct ComponentPartStream {
    name: String,
    component: Component,
    schema: SchemaRef,
}

impl ComponentPartStream {
    pub fn new(name: String, component: Component) -> Self {
        let record_batch = component.as_record_batch(&name);
        let schema = record_batch.schema();
        Self {
            name,
            component,
            schema,
        }
    }
}

impl std::fmt::Debug for ComponentPartStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentPartStream")
            .field("name", &self.name)
            .field("schema", &self.schema)
            .finish()
    }
}

impl std::fmt::Debug for ComponentStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentStream").finish()
    }
}

pub fn component_stream(
    component: Component,
    name: impl ToString,
) -> impl futures_lite::Stream<Item = Result<RecordBatch, datafusion::error::DataFusionError>> {
    let name = name.to_string();
    let current_len = component.time_series.index().len() as usize / size_of::<Timestamp>();
    futures_lite::stream::unfold(
        (component, current_len, name),
        |(component, last_pos, name)| async move {
            let waiter = component.time_series.waiter();
            let _ = waiter.wait().await;

            let current_len = component.time_series.index().len() as usize / size_of::<Timestamp>();
            let record_batch = component.as_record_batch_range(name.clone(), last_pos..current_len);

            Some((Ok(record_batch), (component, current_len, name)))
        },
    )
}

impl Stream for ComponentStream {
    type Item = Result<RecordBatch, datafusion::error::DataFusionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        this.stream.poll_next(cx)
    }
}

impl RecordBatchStream for ComponentStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl PartitionStream for ComponentPartStream {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(
        &self,
        _ctx: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::execution::SendableRecordBatchStream {
        Box::pin(ComponentStream {
            stream: Box::pin(component_stream(self.component.clone(), self.name.clone())),
            schema: self.schema.clone(),
        })
    }
}
