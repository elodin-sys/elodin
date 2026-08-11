use std::{borrow::Cow, ops::Range, sync::Arc};

use arrow::ipc::writer::StreamWriter;
use futures_lite::StreamExt;
use impeller2::types::{ComponentId, PrimType as DbPrimType, Timestamp};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use super::{
    common,
    v1::{
        self, DumpMetadataRequest, DumpMetadataResponse, DumpSchemaRequest, DumpSchemaResponse,
        GetTimeRangeRequest, GetTimeRangeResponse, GetTimeSeriesRequest, GetTimeSeriesResponse,
        SqlRequest, SqlResponse, get_time_series_response, query_service_server::QueryService,
    },
};
use crate::{
    DB,
    arrow::lttb::{DataPoint, lttb_downsample},
};

const CHUNK_BYTES: usize = 1024 * 1024;
type Downsampled<'a> = (Cow<'a, [Timestamp]>, Cow<'a, [u8]>);

#[derive(Clone)]
pub(super) struct QueryServiceImpl {
    db: Arc<DB>,
}

impl QueryServiceImpl {
    pub(super) fn new(db: Arc<DB>) -> Self {
        Self { db }
    }
}

#[tonic::async_trait]
impl QueryService for QueryServiceImpl {
    type GetTimeSeriesStream = ReceiverStream<Result<GetTimeSeriesResponse, Status>>;
    type SqlStream = ReceiverStream<Result<SqlResponse, Status>>;

    async fn get_time_range(
        &self,
        _request: Request<GetTimeRangeRequest>,
    ) -> Result<Response<GetTimeRangeResponse>, Status> {
        let last = self.db.last_updated.latest();
        Ok(Response::new(GetTimeRangeResponse {
            has_data: last.0 != i64::MIN,
            earliest_ns: self.db.earliest_timestamp.latest().0.saturating_mul(1000),
            last_updated_ns: last.0.saturating_mul(1000),
        }))
    }

    async fn dump_metadata(
        &self,
        _request: Request<DumpMetadataRequest>,
    ) -> Result<Response<DumpMetadataResponse>, Status> {
        let (mut components, messages, config) = self.db.with_state(|state| {
            (
                state
                    .component_metadata
                    .values()
                    .map(common::component_metadata)
                    .collect::<Vec<_>>(),
                state
                    .msg_logs
                    .values()
                    .filter_map(|log| log.metadata())
                    .map(common::message_metadata)
                    .collect::<Result<Vec<_>, _>>(),
                common::db_config(&state.db_config),
            )
        });
        let mut messages = messages?;
        components.sort_by(|a, b| a.name.cmp(&b.name));
        messages.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Response::new(DumpMetadataResponse {
            components,
            messages,
            config: Some(config),
        }))
    }

    async fn dump_schema(
        &self,
        _request: Request<DumpSchemaRequest>,
    ) -> Result<Response<DumpSchemaResponse>, Status> {
        let mut components = self.db.with_state(|state| {
            state
                .components
                .values()
                .map(|component| {
                    let name = state
                        .component_metadata
                        .get(&component.component_id)
                        .map_or_else(
                            || component.component_id.to_string(),
                            |value| value.name.clone(),
                        );
                    v1::ComponentSchemaSnapshot {
                        name,
                        prim_type: common::prim_type(component.schema.prim_type) as i32,
                        dims: component.schema.shape().into_vec(),
                        start_time_ns: component.time_series.index_extra().0.saturating_mul(1000),
                    }
                })
                .collect::<Vec<_>>()
        });
        components.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Response::new(DumpSchemaResponse { components }))
    }

    async fn get_time_series(
        &self,
        request: Request<GetTimeSeriesRequest>,
    ) -> Result<Response<Self::GetTimeSeriesStream>, Status> {
        let request = request.into_inner();
        if request.component.is_empty() {
            return Err(Status::invalid_argument("component must be non-empty"));
        }
        let component_id = ComponentId::new(&request.component);
        let (component, element_names) = self
            .db
            .with_state(|state| {
                let component = state.get_component(component_id)?.clone();
                let element_names = state
                    .get_component_metadata(component_id)
                    .map(|metadata| common::element_names(metadata.element_names()))
                    .unwrap_or_default();
                Some((component, element_names))
            })
            .ok_or_else(|| {
                Status::not_found(format!("component {} not found", request.component))
            })?;

        let end = if request.end_ns == 0 {
            Timestamp(i64::MAX)
        } else {
            Timestamp(request.end_ns / 1000)
        };
        let range = Range {
            start: Timestamp(request.start_ns / 1000),
            end,
        };
        let Some((timestamps, data)) = component.time_series.get_range(&range) else {
            return Err(Status::out_of_range("requested time range has no samples"));
        };
        let row_size = component.schema.size();
        let limit = if request.limit == 0 {
            timestamps.len()
        } else {
            usize::try_from(request.limit)
                .unwrap_or(usize::MAX)
                .min(timestamps.len())
        };
        let timestamps = &timestamps[..limit];
        let data = &data[..limit * row_size];
        let (timestamps, data) = downsample(
            timestamps,
            data,
            row_size,
            component.schema.prim_type,
            component.schema.dim.iter().product(),
            request.element_index as usize,
            request.max_points as usize,
        )?;
        let timestamps = timestamps.into_owned();
        let data = data.into_owned();

        let (tx, rx) = mpsc::channel(16);
        let header = v1::TimeSeriesHeader {
            component: request.component,
            prim_type: common::prim_type(component.schema.prim_type) as i32,
            dims: component.schema.shape().into_vec(),
            element_names,
        };
        tx.send(Ok(GetTimeSeriesResponse {
            chunk: Some(get_time_series_response::Chunk::Header(header)),
        }))
        .await
        .map_err(|_| Status::cancelled("client closed response stream"))?;
        tokio::spawn(async move {
            let rows_per_chunk = (CHUNK_BYTES / row_size.max(1)).max(1);
            for (time_chunk, data_chunk) in timestamps
                .chunks(rows_per_chunk)
                .zip(data.chunks(rows_per_chunk * row_size))
            {
                let response = GetTimeSeriesResponse {
                    chunk: Some(get_time_series_response::Chunk::Data(v1::TimeSeriesData {
                        timestamps_ns: time_chunk
                            .iter()
                            .map(|timestamp| timestamp.0.saturating_mul(1000))
                            .collect(),
                        packed_values: data_chunk.to_vec(),
                    })),
                };
                if tx.send(Ok(response)).await.is_err() {
                    break;
                }
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn sql(&self, request: Request<SqlRequest>) -> Result<Response<Self::SqlStream>, Status> {
        let sql = request.into_inner().sql;
        if sql.trim().is_empty() {
            return Err(Status::invalid_argument("sql must be non-empty"));
        }
        let db = self.db.clone();
        let (tx, rx) = mpsc::channel(4);
        tokio::spawn(async move {
            let result = async {
                let mut context = db.as_session_context().map_err(common::internal)?;
                db.insert_views(&mut context)
                    .await
                    .map_err(common::internal)?;
                let frame = context.sql(&sql).await.map_err(common::internal)?;
                let mut stream = frame.execute_stream().await.map_err(common::internal)?;
                while let Some(batch) = stream.next().await {
                    let batch = batch.map_err(common::internal)?;
                    let mut bytes = Vec::new();
                    let mut writer = StreamWriter::try_new(&mut bytes, batch.schema_ref())
                        .map_err(common::internal)?;
                    writer.write(&batch).map_err(common::internal)?;
                    writer.finish().map_err(common::internal)?;
                    if tx.send(Ok(SqlResponse { ipc: bytes })).await.is_err() {
                        return Ok::<_, Status>(());
                    }
                }
                Ok(())
            }
            .await;
            if let Err(error) = result {
                let _ = tx.send(Err(error)).await;
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

fn downsample<'a>(
    timestamps: &'a [Timestamp],
    data: &'a [u8],
    row_size: usize,
    prim_type: DbPrimType,
    elements: usize,
    element_index: usize,
    max_points: usize,
) -> Result<Downsampled<'a>, Status> {
    if max_points == 0 || timestamps.len() <= max_points || max_points < 3 {
        return Ok((Cow::Borrowed(timestamps), Cow::Borrowed(data)));
    }
    if element_index >= elements {
        return Err(Status::invalid_argument("element_index is out of bounds"));
    }
    let element_size = prim_type.size();
    let points = timestamps
        .iter()
        .enumerate()
        .map(|(index, timestamp)| {
            let offset = index * row_size + element_index * element_size;
            Ok(DataPoint {
                time: timestamp.0,
                value: element_as_f64(prim_type, &data[offset..offset + element_size])?,
            })
        })
        .collect::<Result<Vec<_>, Status>>()?;
    let selected = lttb_downsample(&points, max_points);
    let mut output_timestamps = Vec::with_capacity(selected.len());
    let mut output_data = Vec::with_capacity(selected.len() * row_size);
    let mut cursor = 0;
    for point in selected {
        let relative = points[cursor..]
            .iter()
            .position(|candidate| {
                candidate.time == point.time && candidate.value.to_bits() == point.value.to_bits()
            })
            .ok_or_else(|| Status::internal("downsampled point was not in source data"))?;
        let index = cursor + relative;
        output_timestamps.push(timestamps[index]);
        output_data.extend_from_slice(&data[index * row_size..(index + 1) * row_size]);
        cursor = index + 1;
    }
    Ok((Cow::Owned(output_timestamps), Cow::Owned(output_data)))
}

fn element_as_f64(prim_type: DbPrimType, bytes: &[u8]) -> Result<f64, Status> {
    macro_rules! read {
        ($ty:ty) => {
            <$ty>::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| Status::internal("component value has invalid length"))?,
            ) as f64
        };
    }
    Ok(match prim_type {
        DbPrimType::U8 => bytes[0] as f64,
        DbPrimType::U16 => read!(u16),
        DbPrimType::U32 => read!(u32),
        DbPrimType::U64 => read!(u64),
        DbPrimType::I8 => (bytes[0] as i8) as f64,
        DbPrimType::I16 => read!(i16),
        DbPrimType::I32 => read!(i32),
        DbPrimType::I64 => read!(i64),
        DbPrimType::Bool => {
            if bytes[0] == 0 {
                0.0
            } else {
                1.0
            }
        }
        DbPrimType::F32 => read!(f32),
        DbPrimType::F64 => read!(f64),
    })
}

#[cfg(test)]
mod tests {
    use impeller2::types::PrimType;
    use impeller2_wkt::ComponentMetadata;
    use tempfile::TempDir;

    use super::*;
    use crate::ComponentSchema;

    fn test_db() -> (TempDir, Arc<DB>) {
        let directory = TempDir::new().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let id = ComponentId::new("demo.signal");
        db.with_state_mut(|state| {
            state
                .insert_component(id, ComponentSchema::new(PrimType::F64, &[]), &db.path)
                .unwrap();
            state
                .set_component_metadata(
                    ComponentMetadata {
                        component_id: id,
                        name: "demo.signal".into(),
                        metadata: Default::default(),
                    },
                    &db.path,
                )
                .unwrap();
        });
        for index in 0..8 {
            db.apply_component_row(
                Timestamp(100 + index),
                &[(id, (index as f64).to_le_bytes().to_vec())],
            )
            .unwrap();
        }
        (directory, db)
    }

    #[tokio::test]
    async fn discovery_and_time_series_match_db() {
        let (_directory, db) = test_db();
        let service = QueryServiceImpl::new(db);

        let time_range = service
            .get_time_range(Request::new(GetTimeRangeRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert!(time_range.has_data);
        assert_eq!(time_range.earliest_ns, 100_000);
        assert_eq!(time_range.last_updated_ns, 107_000);

        let metadata = service
            .dump_metadata(Request::new(DumpMetadataRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert!(
            metadata
                .components
                .iter()
                .any(|item| item.name == "demo.signal")
        );

        let schema = service
            .dump_schema(Request::new(DumpSchemaRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert!(
            schema
                .components
                .iter()
                .any(|item| item.name == "demo.signal")
        );

        let mut stream = service
            .get_time_series(Request::new(GetTimeSeriesRequest {
                component: "demo.signal".into(),
                start_ns: 100_000,
                end_ns: 107_000,
                limit: 0,
                max_points: 4,
                element_index: 0,
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(matches!(
            futures_lite::StreamExt::next(&mut stream)
                .await
                .unwrap()
                .unwrap()
                .chunk,
            Some(get_time_series_response::Chunk::Header(_))
        ));
        let data = futures_lite::StreamExt::next(&mut stream)
            .await
            .unwrap()
            .unwrap();
        let Some(get_time_series_response::Chunk::Data(data)) = data.chunk else {
            panic!("expected data");
        };
        assert_eq!(data.timestamps_ns.len(), 4);
        assert_eq!(data.packed_values.len(), 4 * size_of::<f64>());
    }

    #[tokio::test]
    async fn sql_returns_arrow_ipc() {
        let (_directory, db) = test_db();
        let service = QueryServiceImpl::new(db);
        let mut stream = service
            .sql(Request::new(SqlRequest {
                sql: "select * from demo_signal".into(),
            }))
            .await
            .unwrap()
            .into_inner();
        let response = futures_lite::StreamExt::next(&mut stream)
            .await
            .unwrap()
            .unwrap();
        assert!(!response.ipc.is_empty());
    }
}
