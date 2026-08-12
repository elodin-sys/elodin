use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, atomic::Ordering},
    time::Duration,
};

use impeller2::types::{ComponentId, PrimType as DbPrimType, Timestamp};
use impeller2_wkt::{ComponentMetadata, SetDbConfig};
use prost::Message;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use super::{
    common::{self, SessionKey, SessionResume},
    v1::{
        self, ComponentSchemaConflict, IngestRequest, IngestResponse, SessionAccept, SessionOpen,
        SessionReject, TelemetryBatch, WriteAck, component_value::Value, ingest_request,
        ingest_response, ingest_service_server::IngestService, row,
    },
};
use crate::{ComponentRowApplyError, ComponentSchema as DbComponentSchema, DB, Error as DbError};

const MAX_CLIENT_NAME_LEN: usize = 128;
const MAX_CLIENT_INSTANCE_ID_LEN: usize = 128;
const MAX_SCHEMA_NAME_LEN: usize = 256;
const MAX_COMPONENT_ELEMENTS: usize = 1 << 24;
// Half the gRPC message cap so even a single-row batch always fits; larger
// schemas would pass SessionOpen yet be undeliverable.
const MAX_PACKED_SIZE: usize = MAX_GRPC_MESSAGE_SIZE / 2;
const DEFAULT_MAX_UNACKED_ROWS: u32 = 256;
const DEFAULT_MAX_ACK_DELAY_MS: u32 = 100;
const MAX_UNACKED_ROWS: u32 = 1_000_000;
const MAX_ACK_DELAY_MS: u32 = 10_000;
// Bounds resume-file writes for aggressive ack policies (e.g. one ack per
// batch); the widened crash window is absorbed by content deduplication.
const RESUME_PERSIST_INTERVAL: Duration = Duration::from_millis(250);
pub(super) const MAX_GRPC_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

#[derive(Clone)]
pub(super) struct IngestServiceImpl {
    db: Arc<DB>,
    resume: Arc<SessionResume>,
}

#[derive(Clone)]
struct ValidatedComponent {
    id: ComponentId,
    name: String,
    prim_type: DbPrimType,
    dims: Vec<usize>,
    element_names: Vec<String>,
    packed_offset: usize,
    byte_len: usize,
    timestamp_source: bool,
}

impl ValidatedComponent {
    fn db_schema(&self) -> DbComponentSchema {
        DbComponentSchema::new(self.prim_type, &self.dims)
    }
}

#[derive(Clone)]
struct ValidatedMessage {
    name: String,
    encoding: v1::RowEncoding,
    packed_size: usize,
    components: Vec<ValidatedComponent>,
}

struct Session {
    key: SessionKey,
    messages: HashMap<u32, ValidatedMessage>,
    current_seq: u64,
    ack_policy: NormalizedAckPolicy,
    // Rows at or below this open-time watermark may be crash-window replays
    // and are content-deduplicated; later rows always append.
    dedup_below: Timestamp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NormalizedAckPolicy {
    max_unacked_rows: u64,
    max_ack_delay: Duration,
}

enum OpenOutcome {
    Accepted(SessionAccept, Session),
    Rejected(SessionReject),
}

#[derive(Debug)]
struct RowFailure {
    component: String,
    detail: String,
}

enum ApplyRowFailure {
    Row(RowFailure),
    Fatal(Status),
}

impl IngestServiceImpl {
    pub(super) fn new(db: Arc<DB>) -> Self {
        let resume = Arc::new(SessionResume::new(db.path.clone(), ".grpc-ingest-session-"));
        Self { db, resume }
    }

    fn open_session(&self, open: SessionOpen) -> Result<OpenOutcome, Status> {
        validate_bounded_name("client_name", &open.client_name, MAX_CLIENT_NAME_LEN)?;
        if open.client_instance_id.is_empty()
            || open.client_instance_id.len() > MAX_CLIENT_INSTANCE_ID_LEN
        {
            return Err(Status::invalid_argument(format!(
                "client_instance_id must contain 1..={MAX_CLIENT_INSTANCE_ID_LEN} bytes"
            )));
        }
        let ack_policy = normalize_ack_policy(open.ack_policy.as_ref())?;
        if open.schema_fingerprint.len() != 32 {
            return Err(Status::invalid_argument(
                "schema_fingerprint must be exactly 32 bytes",
            ));
        }
        let schema = open
            .schema
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("schema is required"))?;
        let fingerprint = Sha256::digest(schema.encode_to_vec());
        if fingerprint.as_slice() != open.schema_fingerprint {
            return Err(Status::invalid_argument(
                "schema_fingerprint does not match the encoded SchemaSet",
            ));
        }
        let messages = validate_schema(schema)?;

        let reject = self
            .db
            .with_state_mut(|state| {
                let conflicts = messages
                    .iter()
                    .flat_map(|message| &message.components)
                    .filter_map(|component| {
                        let existing = state.get_component(component.id)?;
                        let actual = component.db_schema();
                        (existing.schema != actual).then(|| ComponentSchemaConflict {
                            component: component.name.clone(),
                            expected_prim_type: proto_prim_type(existing.schema.prim_type) as i32,
                            expected_dims: existing.schema.shape().into_vec(),
                            actual_prim_type: proto_prim_type(actual.prim_type) as i32,
                            actual_dims: actual.shape().into_vec(),
                        })
                    })
                    .collect::<Vec<_>>();
                if !conflicts.is_empty() {
                    return Ok(Some(SessionReject {
                        detail: format!("{} component schema conflict(s)", conflicts.len()),
                        conflicts,
                    }));
                }
                if let Some(detail) = messages
                    .iter()
                    .flat_map(|message| &message.components)
                    .find_map(|component| {
                        state
                            .get_component_metadata(component.id)
                            .filter(|metadata| {
                                metadata.name != component.name
                                    && metadata.name != component.id.to_string()
                            })
                            .map(|metadata| {
                                format!(
                                    "component {} hashes to the existing component {}",
                                    component.name, metadata.name
                                )
                            })
                    })
                {
                    return Ok(Some(SessionReject {
                        detail,
                        conflicts: Vec::new(),
                    }));
                }
                for component in messages.iter().flat_map(|message| &message.components) {
                    state.insert_component_with_timestamp_source_flag(
                        component.id,
                        component.db_schema(),
                        component.timestamp_source,
                        &self.db.path,
                    )?;
                    // Merge into existing metadata so reconnects never wipe
                    // user-set keys (units, tags, prior element_names).
                    let existing = state.get_component_metadata(component.id).cloned();
                    let mut metadata = existing
                        .as_ref()
                        .map(|meta| meta.metadata.clone())
                        .unwrap_or_default();
                    if !component.element_names.is_empty() {
                        metadata.insert(
                            "element_names".to_string(),
                            component.element_names.join(","),
                        );
                    }
                    if existing.as_ref().is_some_and(|meta| {
                        meta.name == component.name && meta.metadata == metadata
                    }) {
                        continue;
                    }
                    state.set_component_metadata(
                        ComponentMetadata {
                            component_id: component.id,
                            name: component.name.clone(),
                            metadata,
                        },
                        &self.db.path,
                    )?;
                }
                Ok::<_, DbError>(None)
            })
            .map_err(internal_status)?;
        if let Some(reject) = reject {
            return Ok(OpenOutcome::Rejected(reject));
        }
        self.db.vtable_gen.fetch_add(1, Ordering::SeqCst);

        // Skip the config write on reconnects with an unchanged fingerprint
        // so WatchDb subscribers do not see handshake noise.
        let fingerprint_key = fingerprint_metadata_key(&open.client_name);
        let fingerprint_hex = hex(&open.schema_fingerprint);
        let fingerprint_stored = self.db.with_state(|state| {
            state.db_config.metadata.get(&fingerprint_key) == Some(&fingerprint_hex)
        });
        if !fingerprint_stored {
            self.db
                .apply_set_db_config(SetDbConfig {
                    recording: None,
                    metadata: [(fingerprint_key, fingerprint_hex)].into_iter().collect(),
                })
                .map_err(internal_status)?;
        }

        let key = SessionKey {
            client_name: open.client_name,
            client_instance_id: open.client_instance_id,
        };
        let current_seq = self.resume.get(&key);

        let mut order = (0..messages.len()).collect::<Vec<_>>();
        order.sort_unstable_by(|left, right| messages[*left].name.cmp(&messages[*right].name));
        let mut handles = HashMap::with_capacity(messages.len());
        let mut registered = HashMap::with_capacity(messages.len());
        for (index, message_index) in order.into_iter().enumerate() {
            let handle = u32::try_from(index + 1)
                .map_err(|_| Status::invalid_argument("too many message schemas"))?;
            let message = messages[message_index].clone();
            handles.insert(message.name.clone(), handle);
            registered.insert(handle, message);
        }

        Ok(OpenOutcome::Accepted(
            SessionAccept {
                message_handles: handles,
                resume_from_seq: current_seq,
            },
            Session {
                key,
                messages: registered,
                current_seq,
                ack_policy,
                dedup_below: self.db.last_updated.latest(),
            },
        ))
    }

    fn process_batch(
        &self,
        session: &mut Session,
        batch: TelemetryBatch,
    ) -> Result<Vec<IngestResponse>, Status> {
        if !batch.rows.is_empty() && batch.first_seq == 0 {
            return Err(Status::failed_precondition(
                "sequence numbers must start at 1",
            ));
        }
        if batch.first_seq > session.current_seq.saturating_add(1) {
            return Err(Status::failed_precondition(format!(
                "sequence gap: expected at most {}, received {}",
                session.current_seq.saturating_add(1),
                batch.first_seq
            )));
        }
        if !batch.rows.is_empty() {
            let row_count = u64::try_from(batch.rows.len())
                .map_err(|_| Status::invalid_argument("batch has too many rows"))?;
            batch
                .first_seq
                .checked_add(row_count - 1)
                .ok_or_else(|| Status::invalid_argument("batch sequence range overflows"))?;
        }

        let mut responses = Vec::new();
        for (index, row) in batch.rows.iter().enumerate() {
            let seq = batch.first_seq + index as u64;
            if seq <= session.current_seq {
                continue;
            }
            match self.apply_row(session, row) {
                Ok(()) => {}
                Err(ApplyRowFailure::Row(error)) => {
                    responses.push(response(ingest_response::Resp::Error(v1::RowError {
                        seq,
                        component: error.component,
                        detail: error.detail,
                    })));
                }
                Err(ApplyRowFailure::Fatal(status)) => return Err(status),
            }
            session.current_seq = seq;
        }

        self.resume.remember(&session.key, session.current_seq);
        responses.push(response(ingest_response::Resp::Ack(WriteAck {
            through_seq: session.current_seq,
        })));
        Ok(responses)
    }

    // Persistence failures are non-fatal: the client replays from an older
    // point and replays deduplicate.
    fn persist_resume(&self, key: &SessionKey, through_seq: u64) {
        if let Err(error) = self.resume.persist(key, through_seq) {
            tracing::warn!(?error, "failed to persist gRPC ingest resume state");
        }
    }

    // Persist the resume position (throttled), then send the ack.
    async fn flush_ack(
        &self,
        key: &SessionKey,
        tx: &mpsc::Sender<Result<IngestResponse, Status>>,
        ack: WriteAck,
        last_persist: &mut tokio::time::Instant,
    ) -> bool {
        if last_persist.elapsed() >= RESUME_PERSIST_INTERVAL {
            *last_persist = tokio::time::Instant::now();
            self.persist_resume(key, ack.through_seq);
        }
        queue_ack(tx, ack).await
    }

    async fn run_session(
        self,
        mut incoming: tonic::Streaming<IngestRequest>,
        tx: mpsc::Sender<Result<IngestResponse, Status>>,
        mut session: Session,
    ) {
        let ack_policy = session.ack_policy;
        let mut last_sent_ack = session.current_seq;
        let mut pending_ack: Option<WriteAck> = None;
        let mut ack_deadline: Option<tokio::time::Instant> = None;
        let mut last_persist = tokio::time::Instant::now();

        loop {
            let has_pending_ack = pending_ack.is_some();
            let deadline = ack_deadline.unwrap_or_else(tokio::time::Instant::now);
            tokio::select! {
                biased;
                _ = tokio::time::sleep_until(deadline), if has_pending_ack => {
                    let ack = pending_ack.take().expect("pending ack must have a deadline");
                    last_sent_ack = ack.through_seq;
                    ack_deadline = None;
                    if !self.flush_ack(&session.key, &tx, ack, &mut last_persist).await {
                        return;
                    }
                }
                request = incoming.message() => {
                    let request = match request {
                        Ok(Some(request)) => request,
                        Ok(None) => {
                            self.persist_resume(&session.key, session.current_seq);
                            if let Some(ack) = pending_ack.take() {
                                let _ = queue_ack(&tx, ack).await;
                            }
                            return;
                        }
                        Err(status) => {
                            let _ = tx.send(Err(status)).await;
                            return;
                        }
                    };
                    let batch = match request.req {
                        Some(ingest_request::Req::Batch(batch)) => batch,
                        Some(ingest_request::Req::Open(_)) => {
                            let _ = tx
                                .send(Err(Status::failed_precondition(
                                    "SessionOpen is only valid as the first frame",
                                )))
                                .await;
                            return;
                        }
                        None => {
                            let _ = tx
                                .send(Err(Status::invalid_argument(
                                    "ingest request payload is required",
                                )))
                                .await;
                            return;
                        }
                    };
                    let mut responses = match self.process_batch(&mut session, batch) {
                        Ok(responses) => responses,
                        Err(status) => {
                            let _ = tx.send(Err(status)).await;
                            return;
                        }
                    };
                    let ack = match responses.pop().and_then(|response| response.resp) {
                        Some(ingest_response::Resp::Ack(ack)) => ack,
                        _ => {
                            let _ = tx
                                .send(Err(Status::internal(
                                    "batch processing omitted its cumulative ack",
                                )))
                                .await;
                            return;
                        }
                    };
                    for response in responses {
                        if tx.send(Ok(response)).await.is_err() {
                            return;
                        }
                    }
                    if ack.through_seq <= last_sent_ack {
                        continue;
                    }
                    if pending_ack.is_none() {
                        ack_deadline =
                            Some(tokio::time::Instant::now() + ack_policy.max_ack_delay);
                    }
                    pending_ack = Some(ack);
                    if pending_ack
                        .as_ref()
                        .is_some_and(|ack| {
                            ack.through_seq - last_sent_ack >= ack_policy.max_unacked_rows
                        })
                    {
                        let ack = pending_ack.take().expect("pending ack was just set");
                        last_sent_ack = ack.through_seq;
                        ack_deadline = None;
                        if !self.flush_ack(&session.key, &tx, ack, &mut last_persist).await {
                            return;
                        }
                    }
                }
            }
        }
    }

    fn apply_row(&self, session: &Session, row: &v1::Row) -> Result<(), ApplyRowFailure> {
        let message = session.messages.get(&row.message_handle).ok_or_else(|| {
            ApplyRowFailure::Row(RowFailure {
                component: String::new(),
                detail: format!("unknown message handle {}", row.message_handle),
            })
        })?;
        let values = match (&message.encoding, &row.payload) {
            (v1::RowEncoding::Packed, Some(row::Payload::Packed(packed))) => {
                decode_packed(message, packed).map_err(ApplyRowFailure::Row)?
            }
            (v1::RowEncoding::Typed, Some(row::Payload::Typed(typed))) => {
                decode_typed(message, typed).map_err(ApplyRowFailure::Row)?
            }
            (v1::RowEncoding::Packed, _) => {
                return Err(ApplyRowFailure::Row(RowFailure {
                    component: String::new(),
                    detail: format!("message {} requires a packed payload", message.name),
                }));
            }
            (v1::RowEncoding::Typed, _) => {
                return Err(ApplyRowFailure::Row(RowFailure {
                    component: String::new(),
                    detail: format!("message {} requires a typed payload", message.name),
                }));
            }
            (v1::RowEncoding::Unspecified, _) => {
                return Err(ApplyRowFailure::Fatal(Status::internal(
                    "validated message has unspecified encoding",
                )));
            }
        };

        let mut embedded_timestamp_ns = None;
        let mut embedded_timestamp_component = None;
        for ((_, value), component) in values.iter().zip(&message.components) {
            if !component.timestamp_source {
                continue;
            }
            let actual = match component.prim_type {
                DbPrimType::U64 | DbPrimType::I64 => {
                    i64::from_le_bytes(value.as_slice().try_into().map_err(|_| {
                        ApplyRowFailure::Fatal(Status::internal(
                            "timestamp source has invalid byte length",
                        ))
                    })?)
                }
                _ => {
                    return Err(ApplyRowFailure::Fatal(Status::internal(
                        "validated timestamp source has invalid type",
                    )));
                }
            };
            embedded_timestamp_component = Some(component.name.as_str());
            if embedded_timestamp_ns
                .replace(actual)
                .is_some_and(|prior| prior != actual)
            {
                return Err(ApplyRowFailure::Row(RowFailure {
                    component: component.name.clone(),
                    detail: "message timestamp sources do not match".to_string(),
                }));
            }
        }

        let timestamp_ns = match (embedded_timestamp_ns, row.time_monotonic_ns) {
            (Some(embedded), Some(explicit)) if embedded != explicit => {
                return Err(ApplyRowFailure::Row(RowFailure {
                    component: embedded_timestamp_component.unwrap_or_default().to_string(),
                    detail: "time_monotonic_ns does not match the packed timestamp source"
                        .to_string(),
                }));
            }
            (Some(embedded), _) => embedded,
            (None, Some(explicit)) => explicit,
            (None, None) => {
                return Err(ApplyRowFailure::Row(RowFailure {
                    component: String::new(),
                    detail: "row requires time_monotonic_ns or a timestamp_source component"
                        .to_string(),
                }));
            }
        };
        let timestamp = common::record_timestamp(timestamp_ns);
        self.db
            .apply_component_row(timestamp, &values, timestamp <= session.dedup_below)
            .map_err(|error| match error {
                ComponentRowApplyError::TimeTravel(component_id) => {
                    ApplyRowFailure::Row(RowFailure {
                        component: component_name(message, component_id),
                        detail: format!("timestamp {} is older than existing data", timestamp.0),
                    })
                }
                ComponentRowApplyError::Internal(error) => {
                    ApplyRowFailure::Fatal(internal_status(error))
                }
            })
    }
}

#[tonic::async_trait]
impl IngestService for IngestServiceImpl {
    type IngestStream = ReceiverStream<Result<IngestResponse, Status>>;

    async fn ingest(
        &self,
        request: Request<tonic::Streaming<IngestRequest>>,
    ) -> Result<Response<Self::IngestStream>, Status> {
        let mut incoming = request.into_inner();
        let first = incoming
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("SessionOpen is required"))?;
        let open = match first.req {
            Some(ingest_request::Req::Open(open)) => open,
            _ => {
                return Err(Status::failed_precondition(
                    "the first frame must be SessionOpen",
                ));
            }
        };

        let (tx, rx) = mpsc::channel(32);
        match self.open_session(open)? {
            OpenOutcome::Rejected(reject) => {
                let _ = tx
                    .send(Ok(response(ingest_response::Resp::Reject(reject))))
                    .await;
            }
            OpenOutcome::Accepted(accept, session) => {
                tx.send(Ok(response(ingest_response::Resp::Accept(accept))))
                    .await
                    .map_err(|_| Status::cancelled("client closed response stream"))?;
                let service = self.clone();
                tokio::spawn(service.run_session(incoming, tx, session));
            }
        }
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

fn validate_schema(schema: &v1::SchemaSet) -> Result<Vec<ValidatedMessage>, Status> {
    if schema.messages.is_empty() {
        return Err(Status::invalid_argument(
            "schema must contain at least one message",
        ));
    }
    if u32::try_from(schema.messages.len()).is_err() {
        return Err(Status::invalid_argument("schema has too many messages"));
    }
    let mut message_names = HashSet::with_capacity(schema.messages.len());
    let mut component_names = HashSet::new();
    let mut component_ids = HashMap::new();
    let mut messages = Vec::with_capacity(schema.messages.len());
    for message in &schema.messages {
        validate_bounded_name("message name", &message.name, MAX_SCHEMA_NAME_LEN)?;
        if !message_names.insert(message.name.as_str()) {
            return Err(Status::invalid_argument(format!(
                "duplicate message name {}",
                message.name
            )));
        }
        if message.components.is_empty() {
            return Err(Status::invalid_argument(format!(
                "message {} has no components",
                message.name
            )));
        }
        let encoding = v1::RowEncoding::try_from(message.encoding).map_err(|_| {
            Status::invalid_argument(format!("message {} has invalid encoding", message.name))
        })?;
        if encoding == v1::RowEncoding::Unspecified {
            return Err(Status::invalid_argument(format!(
                "message {} has unspecified encoding",
                message.name
            )));
        }
        let packed_size = message.packed_size as usize;
        match encoding {
            v1::RowEncoding::Packed if packed_size == 0 || packed_size > MAX_PACKED_SIZE => {
                return Err(Status::invalid_argument(format!(
                    "message {} has invalid packed_size",
                    message.name
                )));
            }
            v1::RowEncoding::Typed if packed_size != 0 => {
                return Err(Status::invalid_argument(format!(
                    "typed message {} must have packed_size 0",
                    message.name
                )));
            }
            _ => {}
        }

        let mut components = Vec::with_capacity(message.components.len());
        let mut ranges = Vec::with_capacity(message.components.len());
        for component in &message.components {
            validate_bounded_name("component name", &component.name, MAX_SCHEMA_NAME_LEN)?;
            if !component_names.insert(component.name.as_str()) {
                return Err(Status::invalid_argument(format!(
                    "duplicate component name {}",
                    component.name
                )));
            }
            let id = ComponentId::new(&component.name);
            if let Some(existing_name) = component_ids.insert(id, component.name.as_str()) {
                return Err(Status::invalid_argument(format!(
                    "component names {existing_name} and {} have the same hash id",
                    component.name
                )));
            }
            let proto_prim = v1::PrimType::try_from(component.prim_type).map_err(|_| {
                Status::invalid_argument(format!(
                    "component {} has invalid primitive type",
                    component.name
                ))
            })?;
            let prim_type = db_prim_type(proto_prim).ok_or_else(|| {
                Status::invalid_argument(format!(
                    "component {} has unspecified primitive type",
                    component.name
                ))
            })?;
            let mut dims = Vec::with_capacity(component.dims.len());
            let mut element_count = 1usize;
            for dim in &component.dims {
                let dim = usize::try_from(*dim).map_err(|_| {
                    Status::invalid_argument(format!(
                        "component {} dimension does not fit this platform",
                        component.name
                    ))
                })?;
                if dim == 0 {
                    return Err(Status::invalid_argument(format!(
                        "component {} has a zero dimension",
                        component.name
                    )));
                }
                element_count = element_count.checked_mul(dim).ok_or_else(|| {
                    Status::invalid_argument(format!(
                        "component {} dimensions overflow",
                        component.name
                    ))
                })?;
                if element_count > MAX_COMPONENT_ELEMENTS {
                    return Err(Status::invalid_argument(format!(
                        "component {} has too many elements",
                        component.name
                    )));
                }
                dims.push(dim);
            }
            let byte_len = element_count.checked_mul(prim_type.size()).ok_or_else(|| {
                Status::invalid_argument(format!(
                    "component {} byte size overflows",
                    component.name
                ))
            })?;
            if !component.element_names.is_empty() {
                if component.element_names.len() != element_count {
                    return Err(Status::invalid_argument(format!(
                        "component {} has {} element names for {} elements",
                        component.name,
                        component.element_names.len(),
                        element_count
                    )));
                }
                let mut names = HashSet::with_capacity(component.element_names.len());
                for name in &component.element_names {
                    validate_bounded_name("element name", name, MAX_SCHEMA_NAME_LEN)?;
                    if name.contains(',') {
                        return Err(Status::invalid_argument(format!(
                            "component {} element names cannot contain commas",
                            component.name
                        )));
                    }
                    if !names.insert(name.as_str()) {
                        return Err(Status::invalid_argument(format!(
                            "component {} has duplicate element name {}",
                            component.name, name
                        )));
                    }
                }
            }
            if component.timestamp_source
                && (!component.dims.is_empty()
                    || !matches!(prim_type, DbPrimType::U64 | DbPrimType::I64))
            {
                return Err(Status::invalid_argument(format!(
                    "timestamp source {} must be a scalar U64 or I64",
                    component.name
                )));
            }

            let packed_offset = component.packed_offset as usize;
            match encoding {
                v1::RowEncoding::Packed => {
                    if !packed_offset.is_multiple_of(prim_type.alignment()) {
                        return Err(Status::invalid_argument(format!(
                            "component {} offset {} is not {}-byte aligned",
                            component.name,
                            packed_offset,
                            prim_type.alignment()
                        )));
                    }
                    let end = packed_offset.checked_add(byte_len).ok_or_else(|| {
                        Status::invalid_argument(format!(
                            "component {} packed range overflows",
                            component.name
                        ))
                    })?;
                    if end > packed_size {
                        return Err(Status::invalid_argument(format!(
                            "component {} packed range {}..{} exceeds packed_size {}",
                            component.name, packed_offset, end, packed_size
                        )));
                    }
                    ranges.push((packed_offset, end, component.name.as_str()));
                }
                v1::RowEncoding::Typed if packed_offset != 0 => {
                    return Err(Status::invalid_argument(format!(
                        "typed component {} must have packed_offset 0",
                        component.name
                    )));
                }
                _ => {}
            }
            components.push(ValidatedComponent {
                id,
                name: component.name.clone(),
                prim_type,
                dims,
                element_names: component.element_names.clone(),
                packed_offset,
                byte_len,
                timestamp_source: component.timestamp_source,
            });
        }
        ranges.sort_unstable_by_key(|range| range.0);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(Status::invalid_argument(format!(
                    "packed components {} and {} overlap",
                    pair[0].2, pair[1].2
                )));
            }
        }
        messages.push(ValidatedMessage {
            name: message.name.clone(),
            encoding,
            packed_size,
            components,
        });
    }
    Ok(messages)
}

fn decode_packed(
    message: &ValidatedMessage,
    packed: &[u8],
) -> Result<Vec<(ComponentId, Vec<u8>)>, RowFailure> {
    if packed.len() != message.packed_size {
        return Err(RowFailure {
            component: String::new(),
            detail: format!(
                "message {} packed payload has {} bytes, expected {}",
                message.name,
                packed.len(),
                message.packed_size
            ),
        });
    }
    message
        .components
        .iter()
        .map(|component| {
            let end = component
                .packed_offset
                .checked_add(component.byte_len)
                .ok_or_else(|| RowFailure {
                    component: component.name.clone(),
                    detail: "component packed range overflows".to_string(),
                })?;
            let value = packed
                .get(component.packed_offset..end)
                .ok_or_else(|| RowFailure {
                    component: component.name.clone(),
                    detail: "component packed range is out of bounds".to_string(),
                })?;
            if component.prim_type == DbPrimType::Bool && value.iter().any(|byte| *byte > 1) {
                return Err(RowFailure {
                    component: component.name.clone(),
                    detail: "packed bool values must be 0 or 1".to_string(),
                });
            }
            Ok((component.id, value.to_vec()))
        })
        .collect()
}

fn decode_typed(
    message: &ValidatedMessage,
    typed: &v1::TypedValues,
) -> Result<Vec<(ComponentId, Vec<u8>)>, RowFailure> {
    if typed.values.len() != message.components.len() {
        return Err(RowFailure {
            component: String::new(),
            detail: format!(
                "message {} has {} typed values, expected {}",
                message.name,
                typed.values.len(),
                message.components.len()
            ),
        });
    }
    let mut ordered = vec![None; message.components.len()];
    for value in &typed.values {
        let index = usize::try_from(value.component_index).map_err(|_| RowFailure {
            component: String::new(),
            detail: "component_index does not fit this platform".to_string(),
        })?;
        let component = message.components.get(index).ok_or_else(|| RowFailure {
            component: String::new(),
            detail: format!("component_index {} is out of bounds", value.component_index),
        })?;
        if ordered[index].replace(value).is_some() {
            return Err(RowFailure {
                component: component.name.clone(),
                detail: "duplicate typed component value".to_string(),
            });
        }
    }
    message
        .components
        .iter()
        .zip(ordered)
        .map(|(component, value)| {
            let value = value.ok_or_else(|| RowFailure {
                component: component.name.clone(),
                detail: "missing typed component value".to_string(),
            })?;
            let value = value.value.as_ref().ok_or_else(|| RowFailure {
                component: component.name.clone(),
                detail: "typed component value is missing its payload".to_string(),
            })?;
            encode_typed_component(component, value).map(|bytes| (component.id, bytes))
        })
        .collect()
}

fn encode_typed_component(
    component: &ValidatedComponent,
    value: &Value,
) -> Result<Vec<u8>, RowFailure> {
    let fail = || RowFailure {
        component: component.name.clone(),
        detail: format!(
            "typed value does not match {} {}",
            if component.dims.is_empty() {
                "scalar"
            } else {
                "vector"
            },
            component.prim_type
        ),
    };
    if component.dims.is_empty() {
        return match (component.prim_type, value) {
            (DbPrimType::U8, Value::U64(value)) => {
                Ok(vec![u8::try_from(*value).map_err(|_| fail())?])
            }
            (DbPrimType::U16, Value::U64(value)) => Ok(u16::try_from(*value)
                .map_err(|_| fail())?
                .to_le_bytes()
                .to_vec()),
            (DbPrimType::U32, Value::U64(value)) => Ok(u32::try_from(*value)
                .map_err(|_| fail())?
                .to_le_bytes()
                .to_vec()),
            (DbPrimType::U64, Value::U64(value)) => Ok(value.to_le_bytes().to_vec()),
            (DbPrimType::I8, Value::I64(value)) => {
                Ok(vec![i8::try_from(*value).map_err(|_| fail())? as u8])
            }
            (DbPrimType::I16, Value::I64(value)) => Ok(i16::try_from(*value)
                .map_err(|_| fail())?
                .to_le_bytes()
                .to_vec()),
            (DbPrimType::I32, Value::I64(value)) => Ok(i32::try_from(*value)
                .map_err(|_| fail())?
                .to_le_bytes()
                .to_vec()),
            (DbPrimType::I64, Value::I64(value)) => Ok(value.to_le_bytes().to_vec()),
            (DbPrimType::Bool, Value::B(value)) => Ok(vec![u8::from(*value)]),
            (DbPrimType::F32, Value::F32(value)) => Ok(value.to_le_bytes().to_vec()),
            (DbPrimType::F64, Value::F64(value)) => Ok(value.to_le_bytes().to_vec()),
            _ => Err(fail()),
        };
    }

    let expected_len = component
        .byte_len
        .checked_div(component.prim_type.size())
        .ok_or_else(fail)?;
    match (component.prim_type, value) {
        (DbPrimType::U8, Value::U64s(values)) if values.v.len() == expected_len => values
            .v
            .iter()
            .map(|value| u8::try_from(*value).map_err(|_| fail()))
            .collect(),
        (DbPrimType::U16, Value::U64s(values)) if values.v.len() == expected_len => {
            encode_unsigned_array::<u16>(&values.v, component)
        }
        (DbPrimType::U32, Value::U64s(values)) if values.v.len() == expected_len => {
            encode_unsigned_array::<u32>(&values.v, component)
        }
        (DbPrimType::U64, Value::U64s(values)) if values.v.len() == expected_len => Ok(values
            .v
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()),
        (DbPrimType::I8, Value::I64s(values)) if values.v.len() == expected_len => values
            .v
            .iter()
            .map(|value| {
                i8::try_from(*value)
                    .map(|value| value as u8)
                    .map_err(|_| fail())
            })
            .collect(),
        (DbPrimType::I16, Value::I64s(values)) if values.v.len() == expected_len => {
            encode_signed_array::<i16>(&values.v, component)
        }
        (DbPrimType::I32, Value::I64s(values)) if values.v.len() == expected_len => {
            encode_signed_array::<i32>(&values.v, component)
        }
        (DbPrimType::I64, Value::I64s(values)) if values.v.len() == expected_len => Ok(values
            .v
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()),
        (DbPrimType::Bool, Value::Bools(values)) if values.v.len() == expected_len => {
            Ok(values.v.iter().copied().map(u8::from).collect())
        }
        (DbPrimType::F32, Value::F32s(values)) if values.v.len() == expected_len => Ok(values
            .v
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()),
        (DbPrimType::F64, Value::F64s(values)) if values.v.len() == expected_len => Ok(values
            .v
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()),
        _ => Err(fail()),
    }
}

trait LeBytes {
    fn append_le_bytes(self, output: &mut Vec<u8>);
}

macro_rules! impl_le_bytes {
    ($($ty:ty),+) => {
        $(
            impl LeBytes for $ty {
                fn append_le_bytes(self, output: &mut Vec<u8>) {
                    output.extend_from_slice(&self.to_le_bytes());
                }
            }
        )+
    };
}

impl_le_bytes!(u16, u32, i16, i32);

fn encode_unsigned_array<T>(
    values: &[u64],
    component: &ValidatedComponent,
) -> Result<Vec<u8>, RowFailure>
where
    T: TryFrom<u64> + LeBytes,
{
    let mut output = Vec::with_capacity(component.byte_len);
    for value in values {
        T::try_from(*value)
            .map_err(|_| RowFailure {
                component: component.name.clone(),
                detail: format!("typed value is out of range for {}", component.prim_type),
            })?
            .append_le_bytes(&mut output);
    }
    Ok(output)
}

fn encode_signed_array<T>(
    values: &[i64],
    component: &ValidatedComponent,
) -> Result<Vec<u8>, RowFailure>
where
    T: TryFrom<i64> + LeBytes,
{
    let mut output = Vec::with_capacity(component.byte_len);
    for value in values {
        T::try_from(*value)
            .map_err(|_| RowFailure {
                component: component.name.clone(),
                detail: format!("typed value is out of range for {}", component.prim_type),
            })?
            .append_le_bytes(&mut output);
    }
    Ok(output)
}

fn validate_bounded_name(field: &str, value: &str, max_len: usize) -> Result<(), Status> {
    if value.trim().is_empty() || value.len() > max_len || value.chars().any(char::is_control) {
        return Err(Status::invalid_argument(format!(
            "{field} must contain 1..={max_len} non-control UTF-8 bytes"
        )));
    }
    Ok(())
}

fn normalize_ack_policy(policy: Option<&v1::AckPolicy>) -> Result<NormalizedAckPolicy, Status> {
    let max_unacked_rows = match policy.map_or(0, |policy| policy.max_unacked_rows) {
        0 => DEFAULT_MAX_UNACKED_ROWS,
        value => value,
    };
    if !(1..=MAX_UNACKED_ROWS).contains(&max_unacked_rows) {
        return Err(Status::invalid_argument(format!(
            "ack_policy.max_unacked_rows must be zero or 1..={MAX_UNACKED_ROWS}"
        )));
    }
    let max_ack_delay_ms = match policy.map_or(0, |policy| policy.max_ack_delay_ms) {
        0 => DEFAULT_MAX_ACK_DELAY_MS,
        value => value,
    };
    if !(1..=MAX_ACK_DELAY_MS).contains(&max_ack_delay_ms) {
        return Err(Status::invalid_argument(format!(
            "ack_policy.max_ack_delay_ms must be zero or 1..={MAX_ACK_DELAY_MS}"
        )));
    }
    Ok(NormalizedAckPolicy {
        max_unacked_rows: u64::from(max_unacked_rows),
        max_ack_delay: Duration::from_millis(u64::from(max_ack_delay_ms)),
    })
}

async fn queue_ack(tx: &mpsc::Sender<Result<IngestResponse, Status>>, ack: WriteAck) -> bool {
    tx.send(Ok(response(ingest_response::Resp::Ack(ack))))
        .await
        .is_ok()
}

fn db_prim_type(prim_type: v1::PrimType) -> Option<DbPrimType> {
    Some(match prim_type {
        v1::PrimType::Unspecified => return None,
        v1::PrimType::U8 => DbPrimType::U8,
        v1::PrimType::U16 => DbPrimType::U16,
        v1::PrimType::U32 => DbPrimType::U32,
        v1::PrimType::U64 => DbPrimType::U64,
        v1::PrimType::I8 => DbPrimType::I8,
        v1::PrimType::I16 => DbPrimType::I16,
        v1::PrimType::I32 => DbPrimType::I32,
        v1::PrimType::I64 => DbPrimType::I64,
        v1::PrimType::Bool => DbPrimType::Bool,
        v1::PrimType::F32 => DbPrimType::F32,
        v1::PrimType::F64 => DbPrimType::F64,
    })
}

fn proto_prim_type(prim_type: DbPrimType) -> v1::PrimType {
    match prim_type {
        DbPrimType::U8 => v1::PrimType::U8,
        DbPrimType::U16 => v1::PrimType::U16,
        DbPrimType::U32 => v1::PrimType::U32,
        DbPrimType::U64 => v1::PrimType::U64,
        DbPrimType::I8 => v1::PrimType::I8,
        DbPrimType::I16 => v1::PrimType::I16,
        DbPrimType::I32 => v1::PrimType::I32,
        DbPrimType::I64 => v1::PrimType::I64,
        DbPrimType::Bool => v1::PrimType::Bool,
        DbPrimType::F32 => v1::PrimType::F32,
        DbPrimType::F64 => v1::PrimType::F64,
    }
}

fn component_name(message: &ValidatedMessage, component_id: ComponentId) -> String {
    message
        .components
        .iter()
        .find(|component| component.id == component_id)
        .map(|component| component.name.clone())
        .unwrap_or_else(|| component_id.to_string())
}

fn response(resp: ingest_response::Resp) -> IngestResponse {
    IngestResponse { resp: Some(resp) }
}

fn fingerprint_metadata_key(client_name: &str) -> String {
    format!(
        "grpc.schema_fingerprint.{}",
        hex(Sha256::digest(client_name.as_bytes()).as_slice())
    )
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn internal_status(error: DbError) -> Status {
    Status::internal(format!("database ingest failed: {error}"))
}

#[cfg(test)]
mod tests {
    use std::{fs, net::SocketAddr, path::Path};

    use tempfile::TempDir;

    use super::*;
    use v1::{
        BoolArray, ComponentSchema, ComponentValue, DoubleArray, FloatArray, MessageSchema,
        PrimType, Row, RowEncoding, SchemaSet, Sint64Array, TypedValues, Uint64Array,
        ingest_service_server::IngestServiceServer,
    };

    async fn serve(addr: SocketAddr, db: Arc<DB>) -> Result<(), tonic::transport::Error> {
        tonic::transport::Server::builder()
            .add_service(
                IngestServiceServer::new(IngestServiceImpl::new(db))
                    .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE),
            )
            .serve(addr)
            .await
    }

    fn test_db() -> (TempDir, Arc<DB>) {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(DB::create(dir.path().to_path_buf()).unwrap());
        (dir, db)
    }

    type TransportClient =
        v1::ingest_service_client::IngestServiceClient<tonic::transport::Channel>;

    struct TransportHarness {
        addr: SocketAddr,
        _dir: TempDir,
        _db: Arc<DB>,
        server: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
    }

    impl TransportHarness {
        async fn start() -> (Self, TransportClient) {
            let (dir, db) = test_db();
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            drop(listener);
            let server = tokio::spawn(serve(addr, db.clone()));
            let harness = Self {
                addr,
                _dir: dir,
                _db: db,
                server,
            };
            let client = harness.connect().await;
            (harness, client)
        }

        async fn connect(&self) -> TransportClient {
            let endpoint = format!("http://{}", self.addr);
            let mut last_error = None;
            for _ in 0..100 {
                match v1::ingest_service_client::IngestServiceClient::connect(endpoint.clone())
                    .await
                {
                    Ok(client) => return client,
                    Err(error) => last_error = Some(error),
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            panic!("gRPC test server did not start: {}", last_error.unwrap());
        }
    }

    impl Drop for TransportHarness {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    fn open_request(open: SessionOpen) -> IngestRequest {
        IngestRequest {
            req: Some(ingest_request::Req::Open(open)),
        }
    }

    fn batch_request(first_seq: u64, rows: Vec<Row>) -> IngestRequest {
        IngestRequest {
            req: Some(ingest_request::Req::Batch(TelemetryBatch {
                first_seq,
                rows,
            })),
        }
    }

    async fn next_response(responses: &mut tonic::Streaming<IngestResponse>) -> IngestResponse {
        tokio::time::timeout(Duration::from_secs(2), responses.message())
            .await
            .expect("timed out waiting for gRPC ingest response")
            .unwrap()
            .expect("gRPC ingest response stream closed")
    }

    async fn open_transport_session(
        client: &mut TransportClient,
        open: SessionOpen,
    ) -> (
        mpsc::Sender<IngestRequest>,
        tonic::Streaming<IngestResponse>,
        SessionAccept,
    ) {
        let (tx, rx) = mpsc::channel(128);
        tx.send(open_request(open)).await.unwrap();
        let mut responses = client
            .ingest(ReceiverStream::new(rx))
            .await
            .unwrap()
            .into_inner();
        let accept = match next_response(&mut responses).await.resp {
            Some(ingest_response::Resp::Accept(accept)) => accept,
            response => panic!("unexpected handshake response: {response:?}"),
        };
        (tx, responses, accept)
    }

    fn component(
        name: &str,
        prim_type: PrimType,
        dims: &[u64],
        packed_offset: u32,
        timestamp_source: bool,
    ) -> ComponentSchema {
        ComponentSchema {
            name: name.to_string(),
            prim_type: prim_type as i32,
            dims: dims.to_vec(),
            element_names: if dims.is_empty() {
                vec![name.rsplit('.').next().unwrap().to_string()]
            } else {
                (0..dims.iter().product())
                    .map(|index| index.to_string())
                    .collect()
            },
            packed_offset,
            timestamp_source,
        }
    }

    fn packed_schema() -> SchemaSet {
        SchemaSet {
            messages: vec![MessageSchema {
                name: "PackedMessage".to_string(),
                encoding: RowEncoding::Packed as i32,
                packed_size: 16,
                components: vec![
                    component("PACKED.TIME", PrimType::U64, &[], 0, true),
                    component("PACKED.VEC", PrimType::F32, &[2], 8, false),
                ],
            }],
        }
    }

    fn typed_schema() -> SchemaSet {
        SchemaSet {
            messages: vec![MessageSchema {
                name: "TypedMessage".to_string(),
                encoding: RowEncoding::Typed as i32,
                packed_size: 0,
                components: vec![
                    component("TYPED.TIME", PrimType::I64, &[], 0, true),
                    component("TYPED.FLAGS", PrimType::Bool, &[2], 0, false),
                ],
            }],
        }
    }

    fn open(client: &str, instance: &[u8], schema: SchemaSet) -> SessionOpen {
        let schema_fingerprint = Sha256::digest(schema.encode_to_vec()).to_vec();
        SessionOpen {
            client_name: client.to_string(),
            schema_fingerprint,
            schema: Some(schema),
            ack_policy: None,
            client_instance_id: instance.to_vec(),
        }
    }

    fn open_with_ack_policy(
        client: &str,
        instance: &[u8],
        max_unacked_rows: u32,
        max_ack_delay_ms: u32,
    ) -> SessionOpen {
        let mut open = open(client, instance, packed_schema());
        open.ack_policy = Some(v1::AckPolicy {
            max_unacked_rows,
            max_ack_delay_ms,
        });
        open
    }

    fn accepted(
        service: &IngestServiceImpl,
        client: &str,
        instance: &[u8],
        schema: SchemaSet,
    ) -> (SessionAccept, Session) {
        match service
            .open_session(open(client, instance, schema))
            .unwrap()
        {
            OpenOutcome::Accepted(accept, session) => (accept, session),
            OpenOutcome::Rejected(reject) => panic!("unexpected rejection: {}", reject.detail),
        }
    }

    fn packed_row(handle: u32, time_ns: i64, values: [f32; 2]) -> Row {
        let mut packed = time_ns.to_le_bytes().to_vec();
        for value in values {
            packed.extend_from_slice(&value.to_le_bytes());
        }
        Row {
            message_handle: handle,
            time_monotonic_ns: Some(time_ns),
            payload: Some(row::Payload::Packed(packed)),
        }
    }

    fn typed_row(handle: u32, time_ns: i64, flags: [bool; 2]) -> Row {
        Row {
            message_handle: handle,
            time_monotonic_ns: Some(time_ns),
            payload: Some(row::Payload::Typed(TypedValues {
                values: vec![
                    ComponentValue {
                        component_index: 0,
                        value: Some(Value::I64(time_ns)),
                    },
                    ComponentValue {
                        component_index: 1,
                        value: Some(Value::Bools(BoolArray { v: flags.to_vec() })),
                    },
                ],
            })),
        }
    }

    fn codec_component(prim_type: DbPrimType, vector: bool) -> ValidatedComponent {
        ValidatedComponent {
            id: ComponentId(1),
            name: format!("{prim_type}"),
            prim_type,
            dims: if vector { vec![2] } else { Vec::new() },
            element_names: Vec::new(),
            packed_offset: 0,
            byte_len: prim_type.size() * if vector { 2 } else { 1 },
            timestamp_source: false,
        }
    }

    fn ack(response: &IngestResponse) -> Option<u64> {
        match response.resp.as_ref() {
            Some(ingest_response::Resp::Ack(ack)) => Some(ack.through_seq),
            _ => None,
        }
    }

    fn is_row_error(response: &IngestResponse) -> bool {
        matches!(
            response.resp.as_ref(),
            Some(ingest_response::Resp::Error(_))
        )
    }

    #[test]
    fn typed_codec_supports_every_scalar_primitive() {
        let cases = vec![
            (DbPrimType::U8, Value::U64(255), vec![255]),
            (
                DbPrimType::U16,
                Value::U64(0x1234),
                0x1234u16.to_le_bytes().to_vec(),
            ),
            (
                DbPrimType::U32,
                Value::U64(0x12345678),
                0x12345678u32.to_le_bytes().to_vec(),
            ),
            (
                DbPrimType::U64,
                Value::U64(u64::MAX),
                u64::MAX.to_le_bytes().to_vec(),
            ),
            (DbPrimType::I8, Value::I64(-1), vec![0xff]),
            (
                DbPrimType::I16,
                Value::I64(-1234),
                (-1234i16).to_le_bytes().to_vec(),
            ),
            (
                DbPrimType::I32,
                Value::I64(-123456),
                (-123456i32).to_le_bytes().to_vec(),
            ),
            (
                DbPrimType::I64,
                Value::I64(i64::MIN),
                i64::MIN.to_le_bytes().to_vec(),
            ),
            (DbPrimType::Bool, Value::B(true), vec![1]),
            (
                DbPrimType::F32,
                Value::F32(1.25),
                1.25f32.to_le_bytes().to_vec(),
            ),
            (
                DbPrimType::F64,
                Value::F64(-2.5),
                (-2.5f64).to_le_bytes().to_vec(),
            ),
        ];
        for (prim_type, value, expected) in cases {
            assert_eq!(
                encode_typed_component(&codec_component(prim_type, false), &value).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn typed_codec_supports_every_vector_primitive() {
        let cases = vec![
            (
                DbPrimType::U8,
                Value::U64s(Uint64Array { v: vec![1, 2] }),
                vec![1, 2],
            ),
            (
                DbPrimType::U16,
                Value::U64s(Uint64Array { v: vec![1, 2] }),
                [1u16.to_le_bytes(), 2u16.to_le_bytes()].concat(),
            ),
            (
                DbPrimType::U32,
                Value::U64s(Uint64Array { v: vec![1, 2] }),
                [1u32.to_le_bytes(), 2u32.to_le_bytes()].concat(),
            ),
            (
                DbPrimType::U64,
                Value::U64s(Uint64Array { v: vec![1, 2] }),
                [1u64.to_le_bytes(), 2u64.to_le_bytes()].concat(),
            ),
            (
                DbPrimType::I8,
                Value::I64s(Sint64Array { v: vec![-1, 2] }),
                vec![0xff, 2],
            ),
            (
                DbPrimType::I16,
                Value::I64s(Sint64Array { v: vec![-1, 2] }),
                [(-1i16).to_le_bytes(), 2i16.to_le_bytes()].concat(),
            ),
            (
                DbPrimType::I32,
                Value::I64s(Sint64Array { v: vec![-1, 2] }),
                [(-1i32).to_le_bytes(), 2i32.to_le_bytes()].concat(),
            ),
            (
                DbPrimType::I64,
                Value::I64s(Sint64Array { v: vec![-1, 2] }),
                [(-1i64).to_le_bytes(), 2i64.to_le_bytes()].concat(),
            ),
            (
                DbPrimType::Bool,
                Value::Bools(BoolArray {
                    v: vec![true, false],
                }),
                vec![1, 0],
            ),
            (
                DbPrimType::F32,
                Value::F32s(FloatArray { v: vec![1.0, 2.0] }),
                [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat(),
            ),
            (
                DbPrimType::F64,
                Value::F64s(DoubleArray { v: vec![1.0, 2.0] }),
                [1.0f64.to_le_bytes(), 2.0f64.to_le_bytes()].concat(),
            ),
        ];
        for (prim_type, value, expected) in cases {
            assert_eq!(
                encode_typed_component(&codec_component(prim_type, true), &value).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn typed_codec_rejects_overflow_type_and_shape_errors() {
        assert!(
            encode_typed_component(&codec_component(DbPrimType::U8, false), &Value::U64(256))
                .is_err()
        );
        assert!(
            encode_typed_component(&codec_component(DbPrimType::I8, false), &Value::I64(128))
                .is_err()
        );
        assert!(
            encode_typed_component(&codec_component(DbPrimType::U16, false), &Value::I64(1))
                .is_err()
        );
        assert!(
            encode_typed_component(&codec_component(DbPrimType::U16, true), &Value::U64(1))
                .is_err()
        );
        assert!(
            encode_typed_component(
                &codec_component(DbPrimType::U16, true),
                &Value::U64s(Uint64Array { v: vec![1] })
            )
            .is_err()
        );
        let bool_message = ValidatedMessage {
            name: "bools".to_string(),
            encoding: RowEncoding::Packed,
            packed_size: 2,
            components: vec![codec_component(DbPrimType::Bool, true)],
        };
        assert!(decode_packed(&bool_message, &[0, 2]).is_err());
    }

    #[test]
    fn typed_codec_requires_exactly_one_value_per_component() {
        let messages = validate_schema(&typed_schema()).unwrap();
        let message = &messages[0];
        let duplicate = TypedValues {
            values: vec![
                ComponentValue {
                    component_index: 0,
                    value: Some(Value::I64(1)),
                },
                ComponentValue {
                    component_index: 0,
                    value: Some(Value::I64(1)),
                },
            ],
        };
        assert!(decode_typed(message, &duplicate).is_err());
        let missing = TypedValues {
            values: vec![ComponentValue {
                component_index: 0,
                value: Some(Value::I64(1)),
            }],
        };
        assert!(decode_typed(message, &missing).is_err());
    }

    #[test]
    fn handshake_accepts_valid_schema_and_persists_fingerprint() {
        let (_dir, db) = test_db();
        let service = IngestServiceImpl::new(db.clone());
        let schema = packed_schema();
        let fingerprint = Sha256::digest(schema.encode_to_vec()).to_vec();
        let (accept, session) = accepted(&service, "client", b"instance", schema);
        assert_eq!(accept.resume_from_seq, 0);
        assert_eq!(accept.message_handles["PackedMessage"], 1);
        assert_eq!(
            session.ack_policy,
            NormalizedAckPolicy {
                max_unacked_rows: 256,
                max_ack_delay: Duration::from_millis(100),
            }
        );
        let key = fingerprint_metadata_key("client");
        assert_eq!(
            db.with_state(|state| state.db_config.metadata.get(&key).cloned()),
            Some(hex(&fingerprint))
        );
        let metadata = db.with_state(|state| {
            state
                .get_component_metadata(ComponentId::new("PACKED.TIME"))
                .cloned()
                .unwrap()
        });
        assert!(metadata.is_timestamp_source());
        assert_eq!(metadata.element_names(), "TIME");
    }

    #[test]
    fn handshake_normalizes_zero_ack_policy_fields() {
        let (_dir, db) = test_db();
        let service = IngestServiceImpl::new(db);
        let request = open_with_ack_policy("client", b"instance", 0, 0);
        let OpenOutcome::Accepted(_, session) = service.open_session(request).unwrap() else {
            panic!("valid schema was rejected");
        };
        assert_eq!(
            session.ack_policy,
            NormalizedAckPolicy {
                max_unacked_rows: 256,
                max_ack_delay: Duration::from_millis(100),
            }
        );
    }

    #[test]
    fn handshake_rejects_invalid_fingerprint_and_schema() {
        let (_dir, db) = test_db();
        let service = IngestServiceImpl::new(db);
        let mut invalid_fingerprint = open("client", b"instance", packed_schema());
        invalid_fingerprint.schema_fingerprint[0] ^= 1;
        let fingerprint_error = match service.open_session(invalid_fingerprint) {
            Err(error) => error,
            Ok(_) => panic!("invalid fingerprint was accepted"),
        };
        assert_eq!(fingerprint_error.code(), tonic::Code::InvalidArgument);

        let mut schema = packed_schema();
        schema.messages[0].components[0].prim_type = PrimType::F32 as i32;
        let invalid_schema = open("client", b"instance", schema);
        let schema_error = match service.open_session(invalid_schema) {
            Err(error) => error,
            Ok(_) => panic!("invalid schema was accepted"),
        };
        assert_eq!(schema_error.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn handshake_reports_every_schema_conflict_without_mutation() {
        let (_dir, db) = test_db();
        db.with_state_mut(|state| {
            state
                .insert_component(
                    ComponentId::new("PACKED.TIME"),
                    DbComponentSchema::new(DbPrimType::F64, &[]),
                    &db.path,
                )
                .unwrap();
            state
                .insert_component(
                    ComponentId::new("PACKED.VEC"),
                    DbComponentSchema::new(DbPrimType::U16, &[2]),
                    &db.path,
                )
                .unwrap();
        });
        let service = IngestServiceImpl::new(db.clone());
        let outcome = service
            .open_session(open("client", b"instance", packed_schema()))
            .unwrap();
        let OpenOutcome::Rejected(reject) = outcome else {
            panic!("schema conflict was accepted");
        };
        assert_eq!(reject.conflicts.len(), 2);
        assert_eq!(
            db.with_state(|state| state
                .get_component(ComponentId::new("PACKED.TIME"))
                .unwrap()
                .schema
                .prim_type),
            DbPrimType::F64
        );
        let key = fingerprint_metadata_key("client");
        assert!(db.with_state(|state| !state.db_config.metadata.contains_key(&key)));
    }

    #[test]
    fn schema_validation_rejects_duplicate_alignment_bounds_and_overlap() {
        let mut duplicate_message = packed_schema();
        duplicate_message
            .messages
            .push(duplicate_message.messages[0].clone());
        assert!(validate_schema(&duplicate_message).is_err());

        let mut duplicate_component = packed_schema();
        let duplicate = duplicate_component.messages[0].components[1].clone();
        duplicate_component.messages[0].components.push(duplicate);
        assert!(validate_schema(&duplicate_component).is_err());

        let mut misaligned = packed_schema();
        misaligned.messages[0].components[0].packed_offset = 4;
        assert!(validate_schema(&misaligned).is_err());

        let mut out_of_bounds = packed_schema();
        out_of_bounds.messages[0].packed_size = 12;
        assert!(validate_schema(&out_of_bounds).is_err());

        let mut overlap = packed_schema();
        overlap.messages[0].components[1].packed_offset = 4;
        assert!(validate_schema(&overlap).is_err());
    }

    #[test]
    fn packed_and_typed_rows_round_trip() {
        let (_dir, db) = test_db();
        let service = IngestServiceImpl::new(db.clone());
        let mut schema = packed_schema();
        schema.messages.extend(typed_schema().messages);
        let (accept, mut session) = accepted(&service, "client", b"instance", schema);
        let packed_handle = accept.message_handles["PackedMessage"];
        let typed_handle = accept.message_handles["TypedMessage"];

        let packed_time = 1_234_000i64;
        let mut packed = packed_row(packed_handle, packed_time, [1.5, -2.0]);
        packed.time_monotonic_ns = None;
        let responses = service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 1,
                    rows: vec![packed],
                },
            )
            .unwrap();
        assert_eq!(ack(responses.last().unwrap()), Some(1));

        let typed_time = 2_345_000i64;
        let responses = service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 2,
                    rows: vec![typed_row(typed_handle, typed_time, [true, false])],
                },
            )
            .unwrap();
        assert_eq!(ack(responses.last().unwrap()), Some(2));

        db.with_state(|state| {
            assert_eq!(
                state
                    .get_component(ComponentId::new("PACKED.VEC"))
                    .unwrap()
                    .time_series
                    .get(Timestamp(1234))
                    .unwrap(),
                [1.5f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat()
            );
            assert_eq!(
                state
                    .get_component(ComponentId::new("TYPED.FLAGS"))
                    .unwrap()
                    .time_series
                    .get(Timestamp(2345))
                    .unwrap(),
                [1, 0]
            );
        });
        assert_eq!(db.last_updated.latest(), Timestamp(2345));
        assert_eq!(db.earliest_timestamp.latest(), Timestamp(1234));
    }

    #[test]
    fn reconnect_resumes_and_replay_deduplicates() {
        let (_dir, db) = test_db();
        let service = IngestServiceImpl::new(db.clone());
        let schema = packed_schema();
        let (accept, mut session) = accepted(&service, "client", b"instance", schema.clone());
        let handle = accept.message_handles["PackedMessage"];
        let row = packed_row(handle, 1_000_000, [1.0, 2.0]);
        service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 1,
                    rows: vec![row.clone()],
                },
            )
            .unwrap();

        let (accept, mut reconnect) = accepted(&service, "client", b"instance", schema.clone());
        assert_eq!(accept.resume_from_seq, 1);
        let replay = packed_row(
            accept.message_handles["PackedMessage"],
            1_000_000,
            [1.0, 2.0],
        );
        let responses = service
            .process_batch(
                &mut reconnect,
                TelemetryBatch {
                    first_seq: 1,
                    rows: vec![replay],
                },
            )
            .unwrap();
        assert_eq!(ack(responses.last().unwrap()), Some(1));
        assert_eq!(
            db.with_state(|state| state
                .get_component(ComponentId::new("PACKED.VEC"))
                .unwrap()
                .time_series
                .sample_count()),
            1
        );

        let (new_process, _) = accepted(&service, "client", b"new-instance", schema);
        assert_eq!(new_process.resume_from_seq, 0);
    }

    #[test]
    fn distinct_rows_at_one_timestamp_are_preserved() {
        let (_dir, db) = test_db();
        let service = IngestServiceImpl::new(db.clone());
        let (accept, mut session) = accepted(&service, "client", b"instance", packed_schema());
        let handle = accept.message_handles["PackedMessage"];
        service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 1,
                    rows: vec![packed_row(handle, 1_000_000, [1.0, 2.0])],
                },
            )
            .unwrap();
        let responses = service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 2,
                    rows: vec![packed_row(handle, 1_000_000, [3.0, 4.0])],
                },
            )
            .unwrap();
        assert_eq!(ack(responses.last().unwrap()), Some(2));
        assert!(!responses.iter().any(is_row_error));
        assert_eq!(
            db.with_state(|state| state
                .get_component(ComponentId::new("PACKED.VEC"))
                .unwrap()
                .time_series
                .sample_count()),
            2
        );
    }

    #[test]
    fn reconnect_with_same_schema_skips_config_write() {
        let (_dir, db) = test_db();
        let service = IngestServiceImpl::new(db.clone());
        let (_, _) = accepted(&service, "client", b"instance", packed_schema());
        let generation = db.db_config_gen.latest();
        let (_, _) = accepted(&service, "client", b"reconnect", packed_schema());
        assert_eq!(db.db_config_gen.latest(), generation);
    }

    #[test]
    fn session_open_preserves_existing_component_metadata() {
        let (_dir, db) = test_db();
        let service = IngestServiceImpl::new(db.clone());
        let (_, _) = accepted(&service, "client", b"instance", packed_schema());
        let id = ComponentId::new("PACKED.VEC");
        db.with_state_mut(|state| {
            let mut metadata = state.get_component_metadata(id).unwrap().clone();
            metadata
                .metadata
                .insert("unit".to_string(), "m/s".to_string());
            state.set_component_metadata(metadata, &db.path).unwrap();
        });

        let (_, _) = accepted(&service, "client", b"reconnect", packed_schema());
        db.with_state(|state| {
            let metadata = state.get_component_metadata(id).unwrap();
            assert_eq!(
                metadata.metadata.get("unit").map(String::as_str),
                Some("m/s")
            );
            assert!(metadata.metadata.contains_key("element_names"));
        });
    }

    #[test]
    fn identical_rows_with_new_sequences_are_preserved() {
        let (_dir, db) = test_db();
        let service = IngestServiceImpl::new(db.clone());
        let (accept, mut session) = accepted(&service, "client", b"instance", packed_schema());
        let handle = accept.message_handles["PackedMessage"];
        for seq in 1..=2 {
            let responses = service
                .process_batch(
                    &mut session,
                    TelemetryBatch {
                        first_seq: seq,
                        rows: vec![packed_row(handle, 1_000_000, [1.0, 2.0])],
                    },
                )
                .unwrap();
            assert!(!responses.iter().any(is_row_error));
        }
        // Fresh rows above the open-time watermark always append, even when
        // their content matches an existing occurrence.
        assert_eq!(
            db.with_state(|state| state
                .get_component(ComponentId::new("PACKED.VEC"))
                .unwrap()
                .time_series
                .sample_count()),
            2
        );
    }

    #[test]
    fn restart_replay_deduplicates_each_distinct_row_occurrence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        let schema = packed_schema();
        let time_ns = 1_000_000;
        {
            let db = Arc::new(DB::create(path.clone()).unwrap());
            let service = IngestServiceImpl::new(db);
            let (accept, mut session) = accepted(&service, "client", b"instance", schema.clone());
            let handle = accept.message_handles["PackedMessage"];
            service
                .process_batch(
                    &mut session,
                    TelemetryBatch {
                        first_seq: 1,
                        rows: vec![
                            packed_row(handle, time_ns, [1.0, 2.0]),
                            packed_row(handle, time_ns, [3.0, 4.0]),
                        ],
                    },
                )
                .unwrap();
        }

        let db = Arc::new(DB::open(path).unwrap());
        let service = IngestServiceImpl::new(db.clone());
        let (accept, mut session) = accepted(&service, "client", b"instance", schema);
        let handle = accept.message_handles["PackedMessage"];
        let responses = service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 1,
                    rows: vec![
                        packed_row(handle, time_ns, [1.0, 2.0]),
                        packed_row(handle, time_ns, [3.0, 4.0]),
                    ],
                },
            )
            .unwrap();
        assert_eq!(ack(responses.last().unwrap()), Some(2));
        assert!(!responses.iter().any(is_row_error));
        assert_eq!(
            db.with_state(|state| state
                .get_component(ComponentId::new("PACKED.VEC"))
                .unwrap()
                .time_series
                .sample_count()),
            2
        );
    }

    #[test]
    fn timestamp_mismatch_and_time_travel_are_nonfatal() {
        let (_dir, db) = test_db();
        let service = IngestServiceImpl::new(db);
        let (accept, mut session) = accepted(&service, "client", b"instance", packed_schema());
        let handle = accept.message_handles["PackedMessage"];
        let mut mismatch = packed_row(handle, 1_000_000, [1.0, 2.0]);
        let Some(row::Payload::Packed(packed)) = mismatch.payload.as_mut() else {
            unreachable!()
        };
        packed[..8].copy_from_slice(&2_000_000i64.to_le_bytes());
        let responses = service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 1,
                    rows: vec![mismatch],
                },
            )
            .unwrap();
        assert!(is_row_error(&responses[0]));
        assert_eq!(ack(&responses[1]), Some(1));

        service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 2,
                    rows: vec![packed_row(handle, 2_000_000, [1.0, 2.0])],
                },
            )
            .unwrap();
        let responses = service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 3,
                    rows: vec![packed_row(handle, 1_000_000, [1.0, 2.0])],
                },
            )
            .unwrap();
        assert!(is_row_error(&responses[0]));
        assert_eq!(ack(&responses[1]), Some(3));
    }

    #[test]
    fn sequence_gap_is_stream_fatal() {
        let (_dir, db) = test_db();
        let service = IngestServiceImpl::new(db);
        let (accept, mut session) = accepted(&service, "client", b"instance", packed_schema());
        let error = service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 2,
                    rows: vec![packed_row(
                        accept.message_handles["PackedMessage"],
                        1_000_000,
                        [1.0, 2.0],
                    )],
                },
            )
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::FailedPrecondition);
    }

    #[test]
    fn db_restart_replays_and_self_heals_without_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        let schema = packed_schema();
        {
            let db = Arc::new(DB::create(path.clone()).unwrap());
            let service = IngestServiceImpl::new(db);
            let (accept, mut session) = accepted(&service, "client", b"instance", schema.clone());
            service
                .process_batch(
                    &mut session,
                    TelemetryBatch {
                        first_seq: 1,
                        rows: vec![packed_row(
                            accept.message_handles["PackedMessage"],
                            1_000_000,
                            [1.0, 2.0],
                        )],
                    },
                )
                .unwrap();
        }

        let db = Arc::new(DB::open(path).unwrap());
        let service = IngestServiceImpl::new(db.clone());
        let (accept, mut session) = accepted(&service, "client", b"instance", schema);
        assert_eq!(accept.resume_from_seq, 0);
        service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 1,
                    rows: vec![packed_row(
                        accept.message_handles["PackedMessage"],
                        1_000_000,
                        [1.0, 2.0],
                    )],
                },
            )
            .unwrap();
        assert_eq!(
            db.with_state(|state| state
                .get_component(ComponentId::new("PACKED.VEC"))
                .unwrap()
                .time_series
                .sample_count()),
            1
        );
    }

    #[test]
    fn replay_fills_components_missing_from_a_partial_row() {
        let (_dir, db) = test_db();
        let service = IngestServiceImpl::new(db.clone());
        // Register the schema, then simulate a crash that persisted only one
        // component of the row before the client reconnects and replays.
        let (_, _) = accepted(&service, "client", b"instance", packed_schema());
        db.with_state(|state| {
            state
                .get_component(ComponentId::new("PACKED.TIME"))
                .unwrap()
                .time_series
                .push_buf(Timestamp(1000), &1_000_000u64.to_le_bytes())
                .unwrap();
        });
        crate::AtomicTimestampExt::update_max(&db.last_updated, Timestamp(1000));
        let (accept, mut session) = accepted(&service, "client", b"instance", packed_schema());
        service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 1,
                    rows: vec![packed_row(
                        accept.message_handles["PackedMessage"],
                        1_000_000,
                        [1.0, 2.0],
                    )],
                },
            )
            .unwrap();
        db.with_state(|state| {
            for name in ["PACKED.TIME", "PACKED.VEC"] {
                assert_eq!(
                    state
                        .get_component(ComponentId::new(name))
                        .unwrap()
                        .time_series
                        .sample_count(),
                    1
                );
            }
        });
    }

    fn read_append_log_committed(path: &Path) -> Vec<u8> {
        let data = fs::read(path).unwrap();
        let committed_len = u64::from_ne_bytes(data[0..8].try_into().unwrap()) as usize;
        data[16..committed_len].to_vec()
    }

    #[test]
    fn grpc_storage_matches_direct_db_write_bytes() {
        let grpc_dir = tempfile::tempdir().unwrap();
        let direct_dir = tempfile::tempdir().unwrap();
        let grpc_db = Arc::new(DB::create(grpc_dir.path().to_path_buf()).unwrap());
        let direct_db = Arc::new(DB::create(direct_dir.path().to_path_buf()).unwrap());
        let schema = packed_schema();
        let service = IngestServiceImpl::new(grpc_db.clone());
        let (accept, mut session) = accepted(&service, "client", b"instance", schema.clone());
        let row = packed_row(
            accept.message_handles["PackedMessage"],
            1_000_000,
            [1.0, 2.0],
        );
        service
            .process_batch(
                &mut session,
                TelemetryBatch {
                    first_seq: 1,
                    rows: vec![row.clone()],
                },
            )
            .unwrap();

        let validated = validate_schema(&schema).unwrap();
        direct_db.with_state_mut(|state| {
            for component in &validated[0].components {
                state
                    .insert_component_with_timestamp_source_flag(
                        component.id,
                        component.db_schema(),
                        component.timestamp_source,
                        &direct_db.path,
                    )
                    .unwrap();
                let mut metadata = HashMap::new();
                metadata.insert(
                    "element_names".to_string(),
                    component.element_names.join(","),
                );
                state
                    .set_component_metadata(
                        ComponentMetadata {
                            component_id: component.id,
                            name: component.name.clone(),
                            metadata,
                        },
                        &direct_db.path,
                    )
                    .unwrap();
            }
            let row::Payload::Packed(packed) = row.payload.unwrap() else {
                unreachable!()
            };
            for component in &validated[0].components {
                let end = component.packed_offset + component.byte_len;
                state
                    .get_component(component.id)
                    .unwrap()
                    .time_series
                    .push_buf(Timestamp(1000), &packed[component.packed_offset..end])
                    .unwrap();
            }
        });

        for component in &validated[0].components {
            let id = component.id.to_string();
            assert_eq!(
                fs::read(grpc_dir.path().join(&id).join("schema")).unwrap(),
                fs::read(direct_dir.path().join(&id).join("schema")).unwrap()
            );
            for file in ["index", "data"] {
                assert_eq!(
                    read_append_log_committed(&grpc_dir.path().join(&id).join(file)),
                    read_append_log_committed(&direct_dir.path().join(&id).join(file))
                );
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transport_requires_handshake_and_streams_ack() {
        let (_harness, mut client) = TransportHarness::start().await;
        let error = match client
            .ingest(tokio_stream::iter([batch_request(1, Vec::new())]))
            .await
        {
            Err(error) => error,
            Ok(_) => panic!("transport accepted a batch before SessionOpen"),
        };
        assert_eq!(error.code(), tonic::Code::FailedPrecondition);

        let (tx, mut responses, accept) = open_transport_session(
            &mut client,
            open_with_ack_policy("network-client", b"instance", 1, 10_000),
        )
        .await;
        tx.send(batch_request(
            1,
            vec![packed_row(
                accept.message_handles["PackedMessage"],
                1_000_000,
                [1.0, 2.0],
            )],
        ))
        .await
        .unwrap();
        let ack_response = next_response(&mut responses).await;
        assert_eq!(ack(&ack_response), Some(1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transport_coalesces_many_small_batches_by_delay() {
        const BATCH_COUNT: u64 = 64;

        let (_harness, mut client) = TransportHarness::start().await;
        let (tx, mut responses, accept) = open_transport_session(
            &mut client,
            open_with_ack_policy("coalesced-client", b"instance", 1_000_000, 100),
        )
        .await;
        let handle = accept.message_handles["PackedMessage"];
        let started = tokio::time::Instant::now();
        for seq in 1..=BATCH_COUNT {
            tx.send(batch_request(
                seq,
                vec![packed_row(
                    handle,
                    seq as i64 * 1_000_000,
                    [seq as f32, -(seq as f32)],
                )],
            ))
            .await
            .unwrap();
        }

        let mut ack_count = 0;
        let mut through_seq = 0;
        while through_seq < BATCH_COUNT {
            let response = next_response(&mut responses).await;
            match response.resp {
                Some(ingest_response::Resp::Ack(ack)) => {
                    ack_count += 1;
                    through_seq = ack.through_seq;
                }
                response => panic!("unexpected ingest response: {response:?}"),
            }
        }
        let elapsed = started.elapsed();
        eprintln!(
            "ack coalescing: batches={BATCH_COUNT} acks={ack_count} through_seq={through_seq} elapsed_ms={} ack_rate_hz={:.2}",
            elapsed.as_millis(),
            ack_count as f64 / elapsed.as_secs_f64()
        );
        assert!(
            ack_count <= 3,
            "received {ack_count} acks for {BATCH_COUNT} batches"
        );
        assert!(ack_count * 10 < BATCH_COUNT);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transport_flushes_ack_at_row_threshold() {
        let (_harness, mut client) = TransportHarness::start().await;
        let (tx, mut responses, accept) = open_transport_session(
            &mut client,
            open_with_ack_policy("threshold-client", b"instance", 3, 10_000),
        )
        .await;
        let handle = accept.message_handles["PackedMessage"];
        tx.send(batch_request(
            1,
            (1..=3)
                .map(|seq| packed_row(handle, seq * 1_000_000, [seq as f32, 0.0]))
                .collect(),
        ))
        .await
        .unwrap();

        assert_eq!(ack(&next_response(&mut responses).await), Some(3));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transport_flushes_pending_ack_on_idle_deadline() {
        let (_harness, mut client) = TransportHarness::start().await;
        let (tx, mut responses, accept) = open_transport_session(
            &mut client,
            open_with_ack_policy("idle-client", b"instance", 1_000, 100),
        )
        .await;
        let started = tokio::time::Instant::now();
        tx.send(batch_request(
            1,
            vec![packed_row(
                accept.message_handles["PackedMessage"],
                1_000_000,
                [1.0, 2.0],
            )],
        ))
        .await
        .unwrap();

        assert_eq!(ack(&next_response(&mut responses).await), Some(1));
        assert!(started.elapsed() >= Duration::from_millis(75));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transport_flushes_pending_ack_on_half_close() {
        let (_harness, mut client) = TransportHarness::start().await;
        let (tx, mut responses, accept) = open_transport_session(
            &mut client,
            open_with_ack_policy("half-close-client", b"instance", 1_000, 10_000),
        )
        .await;
        tx.send(batch_request(
            1,
            vec![packed_row(
                accept.message_handles["PackedMessage"],
                1_000_000,
                [1.0, 2.0],
            )],
        ))
        .await
        .unwrap();
        drop(tx);

        assert_eq!(ack(&next_response(&mut responses).await), Some(1));
        assert!(
            tokio::time::timeout(Duration::from_secs(2), responses.message())
                .await
                .unwrap()
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transport_queues_row_error_before_covering_ack() {
        let (_harness, mut client) = TransportHarness::start().await;
        let (tx, mut responses, accept) = open_transport_session(
            &mut client,
            open_with_ack_policy("row-error-client", b"instance", 1, 10_000),
        )
        .await;
        let mut row = packed_row(
            accept.message_handles["PackedMessage"],
            1_000_000,
            [1.0, 2.0],
        );
        let Some(row::Payload::Packed(packed)) = row.payload.as_mut() else {
            unreachable!()
        };
        packed[..8].copy_from_slice(&2_000_000i64.to_le_bytes());
        tx.send(batch_request(1, vec![row])).await.unwrap();

        let error = next_response(&mut responses).await;
        let Some(ingest_response::Resp::Error(error)) = error.resp else {
            panic!("expected RowError before WriteAck");
        };
        assert_eq!(error.seq, 1);
        assert_eq!(error.component, "PACKED.TIME");
        assert_eq!(ack(&next_response(&mut responses).await), Some(1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transport_preserves_fatal_status_without_covering_ack() {
        let (_harness, mut client) = TransportHarness::start().await;
        let (tx, mut responses, accept) = open_transport_session(
            &mut client,
            open_with_ack_policy("fatal-client", b"instance", 1_000, 10_000),
        )
        .await;
        let handle = accept.message_handles["PackedMessage"];
        tx.send(batch_request(
            1,
            vec![packed_row(handle, 1_000_000, [1.0, 2.0])],
        ))
        .await
        .unwrap();
        tx.send(batch_request(
            3,
            vec![packed_row(handle, 3_000_000, [3.0, 4.0])],
        ))
        .await
        .unwrap();

        let status = tokio::time::timeout(Duration::from_secs(2), responses.message())
            .await
            .unwrap()
            .unwrap_err();
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transport_rejects_ack_policy_outside_safe_bounds() {
        let (harness, mut client) = TransportHarness::start().await;
        for (max_unacked_rows, max_ack_delay_ms, field) in [
            (1_000_001, 100, "max_unacked_rows"),
            (256, 10_001, "max_ack_delay_ms"),
        ] {
            let error = match client
                .ingest(tokio_stream::iter([open_request(open_with_ack_policy(
                    "invalid-policy-client",
                    b"instance",
                    max_unacked_rows,
                    max_ack_delay_ms,
                ))]))
                .await
            {
                Err(error) => error,
                Ok(_) => panic!("transport accepted invalid {field}"),
            };
            assert_eq!(error.code(), tonic::Code::InvalidArgument);
            assert!(error.message().contains(field));
            client = harness.connect().await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transport_acked_resume_survives_service_restart() {
        let (harness, mut client) = TransportHarness::start().await;
        let (tx, mut responses, accept) = open_transport_session(
            &mut client,
            open_with_ack_policy("restart-client", b"instance", 1, 10_000),
        )
        .await;
        assert_eq!(accept.resume_from_seq, 0);
        let handle = accept.message_handles["PackedMessage"];
        tx.send(batch_request(
            1,
            vec![packed_row(handle, 1_000_000, [1.0, 2.0])],
        ))
        .await
        .unwrap();
        loop {
            if ack(&next_response(&mut responses).await) == Some(1) {
                break;
            }
        }
        drop(tx);
        // Drain to stream end so the session task has persisted its final
        // resume position before the restart below.
        while tokio::time::timeout(Duration::from_secs(2), responses.message())
            .await
            .expect("timed out draining ingest stream")
            .unwrap()
            .is_some()
        {}

        // A fresh service over the same database simulates a restarted
        // process: the resume position persisted at ack time is recovered.
        let restarted = IngestServiceImpl::new(harness._db.clone());
        let (accept, _session) =
            accepted(&restarted, "restart-client", b"instance", packed_schema());
        assert_eq!(accept.resume_from_seq, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transport_reconnect_resumes_cumulative_ack() {
        let (harness, mut client) = TransportHarness::start().await;
        let (tx, mut responses, accept) = open_transport_session(
            &mut client,
            open_with_ack_policy("reconnect-client", b"instance", 1, 10_000),
        )
        .await;
        tx.send(batch_request(
            1,
            vec![packed_row(
                accept.message_handles["PackedMessage"],
                1_000_000,
                [1.0, 2.0],
            )],
        ))
        .await
        .unwrap();
        assert_eq!(ack(&next_response(&mut responses).await), Some(1));
        drop(tx);
        assert!(
            tokio::time::timeout(Duration::from_secs(2), responses.message())
                .await
                .unwrap()
                .unwrap()
                .is_none()
        );

        let mut reconnect = harness.connect().await;
        let (tx, mut responses, accept) = open_transport_session(
            &mut reconnect,
            open_with_ack_policy("reconnect-client", b"instance", 1, 10_000),
        )
        .await;
        assert_eq!(accept.resume_from_seq, 1);
        tx.send(batch_request(
            2,
            vec![packed_row(
                accept.message_handles["PackedMessage"],
                2_000_000,
                [3.0, 4.0],
            )],
        ))
        .await
        .unwrap();
        assert_eq!(ack(&next_response(&mut responses).await), Some(2));
    }
}
