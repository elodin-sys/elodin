use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use impeller2::types::{PacketId, Timestamp, msg_id};
use impeller2_wkt::{LogEntry, MsgMetadata, log_entry_msg_schema, opaque_bytes_msg_schema};
use postcard_schema::schema::owned::OwnedNamedType;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use super::{
    common,
    v1::{
        self, GetMessagesRequest, GetMessagesResponse, PublishAccept, PublishRequest,
        PublishResponse, RegisterRequest, RegisterResponse, WriteAck, get_messages_response,
        message_service_server::MessageService, outgoing_message, publish_request,
        publish_response, register_request,
    },
};
use crate::DB;

#[derive(Clone)]
pub(super) struct MessageServiceImpl {
    db: Arc<DB>,
    resume: Arc<Mutex<HashMap<SessionKey, u64>>>,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct SessionKey {
    client_name: String,
    client_instance_id: Vec<u8>,
}

#[derive(Clone, Copy)]
struct PublishAckPolicy {
    max_unacked: u64,
    max_delay: Duration,
}

impl MessageServiceImpl {
    pub(super) fn new(db: Arc<DB>) -> Self {
        Self {
            db,
            resume: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn metadata(&self, handle: u32) -> Result<(PacketId, MsgMetadata), Status> {
        let id = u16::try_from(handle)
            .map_err(|_| Status::invalid_argument("message_handle is out of range"))?
            .to_le_bytes();
        self.db
            .with_state(|state| {
                state
                    .get_msg_log(id)
                    .and_then(|log| log.metadata().cloned())
            })
            .map(|metadata| (id, metadata))
            .ok_or_else(|| Status::not_found(format!("message handle {handle} is not registered")))
    }

    fn resume_from(&self, key: &SessionKey) -> u64 {
        if let Some(sequence) = self.resume.lock().unwrap().get(key).copied() {
            return sequence;
        }
        let sequence = std::fs::read_to_string(self.session_path(key))
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        self.resume.lock().unwrap().insert(key.clone(), sequence);
        sequence
    }

    fn store_resume(&self, key: &SessionKey, sequence: u64) -> Result<(), std::io::Error> {
        let path = self.session_path(key);
        let directory = path.parent().unwrap();
        std::fs::create_dir_all(directory)?;
        let temporary = path.with_extension("tmp");
        std::fs::write(&temporary, sequence.to_string())?;
        std::fs::rename(temporary, path)?;
        self.resume.lock().unwrap().insert(key.clone(), sequence);
        Ok(())
    }

    fn session_path(&self, key: &SessionKey) -> std::path::PathBuf {
        let mut hash = Sha256::new();
        hash.update(key.client_name.len().to_le_bytes());
        hash.update(key.client_name.as_bytes());
        hash.update(key.client_instance_id.len().to_le_bytes());
        hash.update(&key.client_instance_id);
        self.db
            .path
            .join(format!(".grpc-message-session-{:x}", hash.finalize()))
    }

    fn encode(
        &self,
        metadata: &MsgMetadata,
        payload: Option<outgoing_message::Payload>,
    ) -> Result<Vec<u8>, Status> {
        if metadata.schema == log_entry_msg_schema() {
            let Some(outgoing_message::Payload::Log(log)) = payload else {
                return Err(Status::invalid_argument("log message requires log payload"));
            };
            let level = u8::try_from(log.level)
                .map_err(|_| Status::invalid_argument("log level must fit in uint8"))?;
            return postcard::to_allocvec(&LogEntry {
                level,
                message: log.message,
            })
            .map_err(common::internal);
        }
        let Some(outgoing_message::Payload::Raw(raw)) = payload else {
            return Err(Status::invalid_argument("message requires raw payload"));
        };
        Ok(raw)
    }

    fn process_batch(
        &self,
        batch: v1::PublishBatch,
        key: &SessionKey,
        mut current_seq: u64,
    ) -> Result<(u64, Vec<v1::MessageError>), Status> {
        if batch.messages.is_empty() {
            return Err(Status::invalid_argument("publish batch is empty"));
        }
        if batch.first_seq > current_seq + 1 {
            return Err(Status::failed_precondition(format!(
                "sequence gap: expected {}, got {}",
                current_seq + 1,
                batch.first_seq
            )));
        }
        let mut errors = Vec::new();
        for (index, message) in batch.messages.into_iter().enumerate() {
            let seq = batch.first_seq + index as u64;
            if seq <= current_seq {
                continue;
            }
            let result = self
                .metadata(message.message_handle)
                .and_then(|(id, metadata)| {
                    let payload = self.encode(&metadata, message.payload)?;
                    let timestamp = if message.timestamp_ns == 0 {
                        self.db.apply_implicit_timestamp()
                    } else {
                        Timestamp(message.timestamp_ns / 1000)
                    };
                    self.db
                        .push_msg(timestamp, id, &payload)
                        .map_err(common::db_error)
                });
            if let Err(error) = result {
                let name = self
                    .metadata(message.message_handle)
                    .map_or_else(|_| String::new(), |(_, metadata)| metadata.name);
                errors.push(v1::MessageError {
                    seq,
                    message: name,
                    detail: error.message().to_string(),
                });
            }
            current_seq = seq;
        }
        self.store_resume(key, current_seq)
            .map_err(common::internal)?;
        Ok((current_seq, errors))
    }

    async fn run_publish(
        self,
        mut incoming: tonic::Streaming<PublishRequest>,
        tx: mpsc::Sender<Result<PublishResponse, Status>>,
        key: SessionKey,
        mut current_seq: u64,
        policy: PublishAckPolicy,
    ) {
        let mut pending = 0;
        let mut deadline = tokio::time::Instant::now() + policy.max_delay;
        loop {
            let request = if pending == 0 {
                incoming.message().await
            } else {
                match tokio::time::timeout_at(deadline, incoming.message()).await {
                    Ok(result) => result,
                    Err(_) => {
                        if send_ack(&tx, current_seq).await.is_err() {
                            return;
                        }
                        pending = 0;
                        deadline = tokio::time::Instant::now() + policy.max_delay;
                        continue;
                    }
                }
            };
            let request = match request {
                Ok(Some(request)) => request,
                Ok(None) => {
                    if pending != 0 {
                        let _ = send_ack(&tx, current_seq).await;
                    }
                    return;
                }
                Err(error) => {
                    let _ = tx.send(Err(error)).await;
                    return;
                }
            };
            let Some(publish_request::Request::Batch(batch)) = request.request else {
                let _ = tx
                    .send(Err(Status::invalid_argument(
                        "PublishOpen is only allowed as the first request",
                    )))
                    .await;
                return;
            };
            let row_count = batch.messages.len() as u64;
            let errors = match self.process_batch(batch, &key, current_seq) {
                Ok((next, errors)) => {
                    current_seq = next;
                    errors
                }
                Err(error) => {
                    let _ = tx.send(Err(error)).await;
                    return;
                }
            };
            for error in errors {
                if tx
                    .send(Ok(PublishResponse {
                        response: Some(publish_response::Response::Error(error)),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            if pending == 0 {
                deadline = tokio::time::Instant::now() + policy.max_delay;
            }
            pending += row_count;
            if pending >= policy.max_unacked {
                if send_ack(&tx, current_seq).await.is_err() {
                    return;
                }
                pending = 0;
                deadline = tokio::time::Instant::now() + policy.max_delay;
            }
        }
    }
}

fn normalize_ack_policy(policy: Option<&v1::AckPolicy>) -> Result<PublishAckPolicy, Status> {
    let max_unacked = policy.map_or(256, |value| match value.max_unacked_rows {
        0 => 256,
        rows => rows,
    });
    let max_delay_ms = policy.map_or(100, |value| match value.max_ack_delay_ms {
        0 => 100,
        delay => delay,
    });
    if max_unacked > 1_000_000 || max_delay_ms > 10_000 {
        return Err(Status::invalid_argument("ack policy exceeds safe bounds"));
    }
    Ok(PublishAckPolicy {
        max_unacked: max_unacked as u64,
        max_delay: Duration::from_millis(max_delay_ms as u64),
    })
}

async fn send_ack(
    tx: &mpsc::Sender<Result<PublishResponse, Status>>,
    through_seq: u64,
) -> Result<(), ()> {
    tx.send(Ok(PublishResponse {
        response: Some(publish_response::Response::Ack(WriteAck { through_seq })),
    }))
    .await
    .map_err(|_| ())
}

#[tonic::async_trait]
impl MessageService for MessageServiceImpl {
    type PublishStream = ReceiverStream<Result<PublishResponse, Status>>;
    type GetMessagesStream = ReceiverStream<Result<GetMessagesResponse, Status>>;

    async fn register(
        &self,
        request: Request<RegisterRequest>,
    ) -> Result<Response<RegisterResponse>, Status> {
        let request = request.into_inner();
        if request.name.is_empty() {
            return Err(Status::invalid_argument("message name must be non-empty"));
        }
        let schema = match request.kind {
            Some(register_request::Kind::Opaque(_)) => opaque_bytes_msg_schema(),
            Some(register_request::Kind::Log(_)) => log_entry_msg_schema(),
            Some(register_request::Kind::PostcardSchema(bytes)) => {
                postcard::from_bytes::<OwnedNamedType>(&bytes)
                    .map_err(|error| Status::invalid_argument(error.to_string()))?
            }
            None => return Err(Status::invalid_argument("message kind is required")),
        };
        let id = msg_id(&request.name);
        if let Some(existing) = self.db.with_state(|state| {
            state
                .get_msg_log(id)
                .and_then(|log| log.metadata().cloned())
        }) && (existing.name != request.name || existing.schema != schema)
        {
            return Err(Status::already_exists(
                "message name conflicts with an existing registration",
            ));
        }
        let metadata = MsgMetadata {
            name: request.name,
            schema,
            metadata: request.metadata,
        };
        self.db
            .with_state_mut(|state| state.set_msg_metadata(id, metadata, &self.db.path))
            .map_err(common::db_error)?;
        Ok(Response::new(RegisterResponse {
            message_handle: u16::from_le_bytes(id) as u32,
        }))
    }

    async fn publish(
        &self,
        request: Request<tonic::Streaming<PublishRequest>>,
    ) -> Result<Response<Self::PublishStream>, Status> {
        let mut incoming = request.into_inner();
        let first = incoming
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("stream requires PublishOpen"))?;
        let Some(publish_request::Request::Open(open)) = first.request else {
            return Err(Status::invalid_argument(
                "first publish request must be PublishOpen",
            ));
        };
        if open.client_name.is_empty() || open.client_instance_id.is_empty() {
            return Err(Status::invalid_argument(
                "client_name and client_instance_id must be non-empty",
            ));
        }
        let policy = normalize_ack_policy(open.ack_policy.as_ref())?;
        let key = SessionKey {
            client_name: open.client_name,
            client_instance_id: open.client_instance_id,
        };
        let resume_from_seq = self.resume_from(&key);
        let (tx, rx) = mpsc::channel(32);
        tx.send(Ok(PublishResponse {
            response: Some(publish_response::Response::Accept(PublishAccept {
                resume_from_seq,
            })),
        }))
        .await
        .map_err(|_| Status::cancelled("client closed response stream"))?;
        tokio::spawn(
            self.clone()
                .run_publish(incoming, tx, key, resume_from_seq, policy),
        );
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn get_messages(
        &self,
        request: Request<GetMessagesRequest>,
    ) -> Result<Response<Self::GetMessagesStream>, Status> {
        let request = request.into_inner();
        if request.name.is_empty() {
            return Err(Status::invalid_argument("message name must be non-empty"));
        }
        let id = msg_id(&request.name);
        let end = if request.end_ns == 0 {
            Timestamp(i64::MAX)
        } else {
            Timestamp(request.end_ns / 1000)
        };
        let (metadata, messages) = self
            .db
            .with_state(|state| {
                let log = state.get_msg_log(id)?;
                let metadata = log.metadata().cloned();
                let limit = if request.limit == 0 {
                    usize::MAX
                } else {
                    usize::try_from(request.limit).unwrap_or(usize::MAX)
                };
                let messages = log
                    .get_range(&(Timestamp(request.start_ns / 1000)..end))
                    .take(limit)
                    .map(|(timestamp, payload)| (timestamp, payload.to_vec()))
                    .collect::<Vec<_>>();
                Some((metadata, messages))
            })
            .ok_or_else(|| Status::not_found(format!("message {} not found", request.name)))?;
        let is_log = metadata.is_some_and(|metadata| metadata.schema == log_entry_msg_schema());
        let (tx, rx) = mpsc::channel(32);
        tokio::spawn(async move {
            for (timestamp, payload) in messages {
                let payload = if is_log {
                    match postcard::from_bytes::<LogEntry>(&payload) {
                        Ok(log) => get_messages_response::Payload::Log(v1::LogPayload {
                            level: log.level as u32,
                            message: log.message,
                        }),
                        Err(error) => {
                            let _ = tx.send(Err(common::internal(error))).await;
                            return;
                        }
                    }
                } else {
                    get_messages_response::Payload::Raw(payload)
                };
                if tx
                    .send(Ok(GetMessagesResponse {
                        timestamp_ns: timestamp.0.saturating_mul(1000),
                        payload: Some(payload),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;
    use v1::message_service_server::MessageServiceServer;

    use super::*;

    async fn transport_client() -> (
        TempDir,
        v1::message_service_client::MessageServiceClient<tonic::transport::Channel>,
        tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
    ) {
        let directory = TempDir::new().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let service = MessageServiceImpl::new(db);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(MessageServiceServer::new(service))
                .serve(addr)
                .await
        });
        let endpoint = format!("http://{addr}");
        let client = loop {
            match v1::message_service_client::MessageServiceClient::connect(endpoint.clone()).await
            {
                Ok(client) => break client,
                Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
            }
        };
        (directory, client, server)
    }

    async fn register_raw(
        client: &mut v1::message_service_client::MessageServiceClient<tonic::transport::Channel>,
    ) -> u32 {
        client
            .register(RegisterRequest {
                name: "transport.raw".into(),
                kind: Some(register_request::Kind::Opaque(v1::OpaqueKind {})),
                metadata: Default::default(),
            })
            .await
            .unwrap()
            .into_inner()
            .message_handle
    }

    #[tokio::test]
    async fn register_publish_read_and_replay_deduplicate() {
        let directory = TempDir::new().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let service = MessageServiceImpl::new(db.clone());
        let handle = service
            .register(Request::new(RegisterRequest {
                name: "demo.log".into(),
                kind: Some(register_request::Kind::Log(v1::LogKind {})),
                metadata: Default::default(),
            }))
            .await
            .unwrap()
            .into_inner()
            .message_handle;
        let key = SessionKey {
            client_name: "test".into(),
            client_instance_id: vec![1],
        };
        let batch = v1::PublishBatch {
            first_seq: 1,
            messages: vec![
                v1::OutgoingMessage {
                    message_handle: handle,
                    timestamp_ns: 100_000,
                    payload: Some(outgoing_message::Payload::Log(v1::LogPayload {
                        level: 2,
                        message: "first".into(),
                    })),
                },
                v1::OutgoingMessage {
                    message_handle: handle,
                    timestamp_ns: 101_000,
                    payload: Some(outgoing_message::Payload::Log(v1::LogPayload {
                        level: 3,
                        message: "second".into(),
                    })),
                },
            ],
        };
        let (current, errors) = service.process_batch(batch.clone(), &key, 0).unwrap();
        assert_eq!(current, 2);
        assert!(errors.is_empty());
        let (current, errors) = service.process_batch(batch.clone(), &key, current).unwrap();
        assert_eq!(current, 2);
        assert!(errors.is_empty());
        let restarted = MessageServiceImpl::new(db.clone());
        assert_eq!(restarted.resume_from(&key), 2);
        let (current, errors) = restarted.process_batch(batch, &key, 2).unwrap();
        assert_eq!(current, 2);
        assert!(errors.is_empty());
        assert_eq!(
            db.with_state(|state| state
                .get_msg_log(msg_id("demo.log"))
                .unwrap()
                .timestamps()
                .len()),
            2
        );

        let mut messages = service
            .get_messages(Request::new(GetMessagesRequest {
                name: "demo.log".into(),
                start_ns: 0,
                end_ns: 0,
                limit: 0,
            }))
            .await
            .unwrap()
            .into_inner();
        let first = futures_lite::StreamExt::next(&mut messages)
            .await
            .unwrap()
            .unwrap();
        let Some(get_messages_response::Payload::Log(first)) = first.payload else {
            panic!("expected decoded log");
        };
        assert_eq!(first.message, "first");
        let second = futures_lite::StreamExt::next(&mut messages)
            .await
            .unwrap()
            .unwrap();
        let Some(get_messages_response::Payload::Log(second)) = second.payload else {
            panic!("expected decoded log");
        };
        assert_eq!(second.message, "second");
    }

    #[tokio::test]
    async fn native_push_is_visible_to_grpc_read() {
        let directory = TempDir::new().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let service = MessageServiceImpl::new(db.clone());
        db.push_msg(Timestamp(7), msg_id("demo.raw"), b"native")
            .unwrap();
        let mut messages = service
            .get_messages(Request::new(GetMessagesRequest {
                name: "demo.raw".into(),
                start_ns: 0,
                end_ns: 0,
                limit: 0,
            }))
            .await
            .unwrap()
            .into_inner();
        let message = futures_lite::StreamExt::next(&mut messages)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            message.payload,
            Some(get_messages_response::Payload::Raw(b"native".to_vec()))
        );
    }

    #[tokio::test]
    async fn transport_starts_ack_delay_with_first_pending_batch() {
        let (_directory, mut client, server) = transport_client().await;
        let handle = register_raw(&mut client).await;
        let (tx, rx) = mpsc::channel(4);
        tx.send(PublishRequest {
            request: Some(publish_request::Request::Open(v1::PublishOpen {
                client_name: "transport".into(),
                client_instance_id: vec![1],
                ack_policy: Some(v1::AckPolicy {
                    max_unacked_rows: 100,
                    max_ack_delay_ms: 200,
                }),
            })),
        })
        .await
        .unwrap();
        let mut responses = client
            .publish(ReceiverStream::new(rx))
            .await
            .unwrap()
            .into_inner();
        assert!(matches!(
            responses.message().await.unwrap().unwrap().response,
            Some(publish_response::Response::Accept(_))
        ));
        tokio::time::sleep(Duration::from_millis(250)).await;
        let started = tokio::time::Instant::now();
        tx.send(PublishRequest {
            request: Some(publish_request::Request::Batch(v1::PublishBatch {
                first_seq: 1,
                messages: vec![v1::OutgoingMessage {
                    message_handle: handle,
                    timestamp_ns: 1_000,
                    payload: Some(outgoing_message::Payload::Raw(vec![1])),
                }],
            })),
        })
        .await
        .unwrap();
        let response = responses.message().await.unwrap().unwrap();
        assert!(matches!(
            response.response,
            Some(publish_response::Response::Ack(_))
        ));
        assert!(started.elapsed() >= Duration::from_millis(150));
        drop(tx);
        server.abort();
    }

    #[tokio::test]
    async fn transport_flushes_pending_ack_on_half_close() {
        let (_directory, mut client, server) = transport_client().await;
        let handle = register_raw(&mut client).await;
        let (tx, rx) = mpsc::channel(4);
        tx.send(PublishRequest {
            request: Some(publish_request::Request::Open(v1::PublishOpen {
                client_name: "transport-close".into(),
                client_instance_id: vec![2],
                ack_policy: Some(v1::AckPolicy {
                    max_unacked_rows: 100,
                    max_ack_delay_ms: 10_000,
                }),
            })),
        })
        .await
        .unwrap();
        tx.send(PublishRequest {
            request: Some(publish_request::Request::Batch(v1::PublishBatch {
                first_seq: 1,
                messages: vec![v1::OutgoingMessage {
                    message_handle: handle,
                    timestamp_ns: 1_000,
                    payload: Some(outgoing_message::Payload::Raw(vec![1])),
                }],
            })),
        })
        .await
        .unwrap();
        drop(tx);
        let mut responses = client
            .publish(ReceiverStream::new(rx))
            .await
            .unwrap()
            .into_inner();
        assert!(matches!(
            responses.message().await.unwrap().unwrap().response,
            Some(publish_response::Response::Accept(_))
        ));
        let response = tokio::time::timeout(Duration::from_secs(1), responses.message())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let Some(publish_response::Response::Ack(ack)) = response.response else {
            panic!("expected ack");
        };
        assert_eq!(ack.through_seq, 1);
        server.abort();
    }

    #[test]
    fn persisted_resume_survives_db_reopen() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("db");
        let db = Arc::new(DB::create(path.clone()).unwrap());
        let service = MessageServiceImpl::new(db.clone());
        let key = SessionKey {
            client_name: "test".into(),
            client_instance_id: vec![1],
        };
        service.store_resume(&key, 7).unwrap();
        drop(service);
        drop(db);
        let reopened = Arc::new(DB::open(path).unwrap());
        assert_eq!(MessageServiceImpl::new(reopened).resume_from(&key), 7);
    }
}
