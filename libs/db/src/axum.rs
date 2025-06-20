use axum::extract::{Json, Path, State};
use axum::routing::{get, post};
use axum::{Router, response::IntoResponse};
use futures_lite::StreamExt;
use impeller2::types::Timestamp;
use impeller2_wkt::{ComponentValue, ErrorResponse, MsgMetadata};
use miette::IntoDiagnostic;
use serde::Serialize;
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::Arc;
use zerocopy::{Immutable, TryFromBytes};

use crate::msg_log::MsgLog;
use crate::{Component, DB, Error};

pub async fn serve(addr: SocketAddr, db: Arc<DB>) -> miette::Result<()> {
    let app = Router::new()
        .route("/component/stream/{component_id}", get(stream))
        .route("/component/{component_id}", post(push_entity_table))
        .route("/msg/stream/{msg_id}", get(stream_msgs))
        .route("/msg/{msg_id}", post(push_msg))
        .with_state(db);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .into_diagnostic()?;
    axum::serve(listener, app).await.into_diagnostic()?;
    Ok(())
}

pub async fn push_msg(
    Path(msg_id): Path<String>,
    db: State<Arc<DB>>,
    body: Json<serde_json::Value>,
) -> Result<impl IntoResponse, Json<ErrorResponse>> {
    let msg_id = impeller2::types::msg_id(&msg_id);
    let msg_log = db
        .with_state(|s| s.msg_logs.get(&msg_id).cloned())
        .ok_or(Error::MsgNotFound(msg_id))
        .map_err(ErrorResponse::from)
        .map_err(Json)?;
    let Some(metadata) = msg_log.metadata() else {
        return Err(Json(ErrorResponse {
            description: "msg lacks a schema".to_string(),
        }));
    };
    let msg = postcard_dyn::to_stdvec_dyn(&metadata.schema, &body.0)
        .map_err(|err| ErrorResponse {
            description: format!("{:?}", err),
        })
        .map_err(Json)?;
    msg_log
        .push(Timestamp::now(), &msg)
        .map_err(ErrorResponse::from)
        .map_err(Json)?;

    Ok(())
}

pub async fn push_entity_table(
    Path(component_id): Path<String>,
    db: State<Arc<DB>>,
    body: Json<ComponentValue>,
) -> Result<impl IntoResponse, Json<ErrorResponse>> {
    let component_id = impeller2::types::ComponentId::new(&component_id);
    let component = db
        .with_state(|s| s.get_component(component_id).cloned())
        .ok_or(Error::ComponentNotFound(component_id))
        .map_err(ErrorResponse::from)
        .map_err(Json)?;
    if component.schema.prim_type != body.prim_type() {
        return Err(Json(ErrorResponse {
            description: "incorrect prim_type for value".to_string(),
        }));
    }
    if &component.schema.dim[..] != body.shape() {
        return Err(Json(ErrorResponse {
            description: "incorrect shape for value".to_string(),
        }));
    }
    component
        .time_series
        .push_buf(Timestamp::now(), body.as_bytes())
        .map_err(ErrorResponse::from)
        .map_err(Json)?;
    Ok(())
}

pub async fn stream_msgs(
    Path(msg_id): Path<String>,
    db: State<Arc<DB>>,
) -> Result<impl IntoResponse, Json<ErrorResponse>> {
    let msg_id = impeller2::types::msg_id(&msg_id);
    let msg_log = db
        .with_state(|s| s.msg_logs.get(&msg_id).cloned())
        .ok_or(Error::MsgNotFound(msg_id))
        .map_err(ErrorResponse::from)
        .map_err(Json)?;
    let Some(metadata) = msg_log.metadata().cloned() else {
        return Err(Json(ErrorResponse {
            description: "msg lacks a schema".to_string(),
        }));
    };
    let stream = msg_stream(msg_log, metadata);
    Ok(axum_streams::StreamBodyAs::json_nl(stream))
}

pub fn msg_stream(
    msg_log: MsgLog,
    metadata: MsgMetadata,
) -> impl futures_lite::Stream<Item = Value> {
    futures_lite::stream::unfold((msg_log, metadata), |(msg_log, metadata)| async move {
        let waiter = msg_log.waiter();
        let _ = waiter.wait().await;
        let (_, buf) = msg_log.latest()?;
        let json = match postcard_dyn::from_slice_dyn(&metadata.schema, buf) {
            Ok(v) => v,

            Err(err) => {
                let err = ErrorResponse {
                    description: format!("{:?}", err),
                };
                serde_json::to_value(&err).expect("failed to serialize error")
            }
        };

        Some((json, (msg_log, metadata)))
    })
}

pub async fn stream(
    Path(component_id): Path<String>,
    db: State<Arc<DB>>,
) -> Result<impl IntoResponse, Json<ErrorResponse>> {
    let component_id = impeller2::types::ComponentId::new(&component_id);
    let component = db
        .with_state(|s| s.get_component(component_id).cloned())
        .ok_or(Error::ComponentNotFound(component_id))
        .map_err(ErrorResponse::from)
        .map_err(Json)?;
    let stream = component_stream(component);
    Ok(axum_streams::StreamBodyAs::json_nl(stream))
}

pub fn component_stream(component: Component) -> impl futures_lite::Stream<Item = Value> {
    futures_lite::stream::try_unfold(component, |component| async move {
        let waiter = component.time_series.waiter();
        let _ = waiter.wait().await;
        let Some((&timestamp, buf)) = component.time_series.latest() else {
            return Ok(None);
        };
        pub fn buf_to_json<T: TryFromBytes + Immutable + Serialize>(
            buf: &[u8],
            shape: &[usize],
            timestamp: Timestamp,
        ) -> Value {
            let data = match <[T]>::try_ref_from_bytes(buf)
                .map_err(impeller2::error::Error::from)
                .map_err(Error::from)
            {
                Ok(d) => d,
                Err(err) => {
                    let err = ErrorResponse::from(err);
                    return serde_json::to_value(&err).expect("failed to serialize error");
                }
            };
            let val = StreamValue {
                timestamp,
                data,
                shape,
            };
            serde_json::to_value(&val).expect("failed to serialize value")
        }
        let shape = &component.schema.dim[..];
        let json = match component.schema.prim_type {
            impeller2::types::PrimType::U8 => buf_to_json::<u8>(buf, shape, timestamp),
            impeller2::types::PrimType::U16 => buf_to_json::<u16>(buf, shape, timestamp),
            impeller2::types::PrimType::U32 => buf_to_json::<u32>(buf, shape, timestamp),
            impeller2::types::PrimType::U64 => buf_to_json::<u64>(buf, shape, timestamp),
            impeller2::types::PrimType::I8 => buf_to_json::<i8>(buf, shape, timestamp),
            impeller2::types::PrimType::I16 => buf_to_json::<i16>(buf, shape, timestamp),
            impeller2::types::PrimType::I32 => buf_to_json::<i32>(buf, shape, timestamp),
            impeller2::types::PrimType::I64 => buf_to_json::<i64>(buf, shape, timestamp),
            impeller2::types::PrimType::Bool => buf_to_json::<bool>(buf, shape, timestamp),
            impeller2::types::PrimType::F32 => buf_to_json::<f32>(buf, shape, timestamp),
            impeller2::types::PrimType::F64 => buf_to_json::<f64>(buf, shape, timestamp),
        };
        Ok::<_, Error>(Some((json, component)))
    })
    .filter_map(|res| match res {
        Ok(r) => Some(r),
        Err(err) => {
            tracing::warn!(?err, "error generating json stream");
            None
        }
    })
}

#[derive(Serialize)]
struct StreamValue<'a, T> {
    timestamp: Timestamp,
    data: &'a [T],
    shape: &'a [usize],
}
