use arrow::{
    array::{
        ArrayRef, ArrowPrimitiveType, BooleanArray, FixedSizeListArray, PrimitiveArray,
        RecordBatch, TimestampMicrosecondArray,
    },
    buffer::{BooleanBuffer, Buffer, ScalarBuffer},
    datatypes::*,
};
use convert_case::{Case, Casing};
use datafusion::{datasource::MemTable, parquet, prelude::SessionContext};
use impeller2::types::PrimType;
use impeller2_wkt::ArchiveFormat;
use std::{fs::File, path::Path, ptr::NonNull, sync::Arc};
use zerocopy::{Immutable, IntoBytes};

use crate::{Component, DB, Error, append_log::AppendLog};

impl<T: IntoBytes + Immutable> AppendLog<T> {
    pub fn as_arrow_buffer(&self) -> Buffer {
        let data = self.data();
        unsafe {
            let ptr = NonNull::new(data.as_ptr() as *mut _).expect("mmap null");
            Buffer::from_custom_allocation(ptr, data.len(), self.raw_mmap().clone())
        }
    }
}

impl Component {
    pub fn as_data_array(&self, name: impl ToString) -> (FieldRef, ArrayRef) {
        let size = self.schema.dim.iter().product::<usize>() as i32;
        let buf = self.time_series.data();
        let array = match self.schema.prim_type {
            PrimType::F64 => array_ref::<Float64Type, _>(buf),
            PrimType::F32 => array_ref::<Float32Type, _>(buf),
            PrimType::U64 => array_ref::<UInt64Type, _>(buf),
            PrimType::U32 => array_ref::<UInt32Type, _>(buf),
            PrimType::U16 => array_ref::<UInt16Type, _>(buf),
            PrimType::U8 => array_ref::<UInt8Type, _>(buf),
            PrimType::I64 => array_ref::<Int64Type, _>(buf),
            PrimType::I32 => array_ref::<Int32Type, _>(buf),
            PrimType::I16 => array_ref::<Int16Type, _>(buf),
            PrimType::I8 => array_ref::<Int8Type, _>(buf),
            PrimType::Bool => bool_ref(buf),
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
        let buf = self.time_series.index();
        let array = scalar_buffer::<TimestampMicrosecondType, _>(buf);
        let array = TimestampMicrosecondArray::new(array, None);
        Arc::new(array)
    }

    pub fn as_record_batch(&self, name: impl ToString) -> RecordBatch {
        let name = name.to_string();
        let (data_field, data_array) = self.as_data_array(name.clone());
        let time_array = self.as_time_series_array();
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
                let mem_table = component.as_mem_table(component_name);
                ctx.register_table(name, Arc::new(mem_table))?;
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
                        writer.flush()?;
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

fn bool_ref<T: IntoBytes + Immutable>(buf: &AppendLog<T>) -> ArrayRef {
    let buffer = buf.as_arrow_buffer();
    let len = buffer.len();
    let buf = BooleanBuffer::new(buffer, 0, len);
    Arc::new(BooleanArray::new(buf, None))
}

fn array_ref<P: ArrowPrimitiveType, T: IntoBytes + Immutable>(buf: &AppendLog<T>) -> ArrayRef {
    let scalar_buffer = scalar_buffer::<P, T>(buf);
    let array = PrimitiveArray::<P>::new(scalar_buffer, None);
    Arc::new(array)
}

fn scalar_buffer<P: ArrowPrimitiveType, T: IntoBytes + Immutable>(
    buf: &AppendLog<T>,
) -> ScalarBuffer<P::Native> {
    let buffer = buf.as_arrow_buffer();
    let len = buffer.len() / size_of::<P::Native>();
    ScalarBuffer::<P::Native>::new(buffer, 0, len)
}
