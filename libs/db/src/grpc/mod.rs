use std::{net::SocketAddr, sync::Arc};

use tonic::{Request, Status, service::interceptor::InterceptedService};

use crate::DB;

pub mod v1 {
    tonic::include_proto!("elodin.db.v1");
}

mod admin;
mod common;
mod ingest;
mod msg;
mod query;
mod stream;

const DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("elodin_db_descriptor");

pub async fn serve(
    addr: SocketAddr,
    db: Arc<DB>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    serve_with_auth(addr, db, None).await
}

pub async fn serve_with_auth(
    addr: SocketAddr,
    db: Arc<DB>,
    auth_token: Option<String>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    router(db, auth_token).await?.serve(addr).await?;
    Ok(())
}

// Serves on a listener the caller bound, letting embedders surface bind
// failures synchronously instead of from a background task.
pub async fn serve_listener(
    listener: std::net::TcpListener,
    db: Arc<DB>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    listener.set_nonblocking(true)?;
    let listener = tokio::net::TcpListener::from_std(listener)?;
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    router(db, None)
        .await?
        .serve_with_incoming(incoming)
        .await?;
    Ok(())
}

async fn router(
    db: Arc<DB>,
    auth_token: Option<String>,
) -> Result<tonic::transport::server::Router, Box<dyn std::error::Error + Send + Sync>> {
    use v1::{
        admin_service_server::AdminServiceServer, ingest_service_server::IngestServiceServer,
        message_service_server::MessageServiceServer, query_service_server::QueryServiceServer,
        stream_service_server::StreamServiceServer,
    };

    let (reporter, health) = tonic_health::server::health_reporter();
    for service in [
        "elodin.db.v1.IngestService",
        "elodin.db.v1.QueryService",
        "elodin.db.v1.StreamService",
        "elodin.db.v1.MessageService",
        "elodin.db.v1.AdminService",
    ] {
        reporter
            .set_service_status(service, tonic_health::ServingStatus::Serving)
            .await;
    }
    let reflection_v1 = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(DESCRIPTOR_SET)
        .register_encoded_file_descriptor_set(tonic_health::pb::FILE_DESCRIPTOR_SET)
        .build_v1()?;
    let reflection_v1alpha = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(DESCRIPTOR_SET)
        .register_encoded_file_descriptor_set(tonic_health::pb::FILE_DESCRIPTOR_SET)
        .build_v1alpha()?;
    let auth_token = Arc::new(auth_token);
    let authenticate = move |request: Request<()>| authorize(request, auth_token.as_deref());
    let ingest = IngestServiceServer::new(ingest::IngestServiceImpl::new(db.clone()))
        .max_decoding_message_size(ingest::MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(ingest::MAX_GRPC_MESSAGE_SIZE);
    let admin = AdminServiceServer::new(admin::AdminServiceImpl::new(db.clone()))
        .max_decoding_message_size(ingest::MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(ingest::MAX_GRPC_MESSAGE_SIZE);
    let message = MessageServiceServer::new(msg::MessageServiceImpl::new(db.clone()))
        .max_decoding_message_size(ingest::MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(ingest::MAX_GRPC_MESSAGE_SIZE);
    let query = QueryServiceServer::new(query::QueryServiceImpl::new(db.clone()))
        .max_decoding_message_size(ingest::MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(ingest::MAX_GRPC_MESSAGE_SIZE);
    let stream = StreamServiceServer::new(stream::StreamServiceImpl::new(db))
        .max_decoding_message_size(ingest::MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(ingest::MAX_GRPC_MESSAGE_SIZE);
    Ok(tonic::transport::Server::builder()
        .add_service(InterceptedService::new(ingest, authenticate.clone()))
        .add_service(InterceptedService::new(admin, authenticate.clone()))
        .add_service(InterceptedService::new(message, authenticate.clone()))
        .add_service(InterceptedService::new(query, authenticate.clone()))
        .add_service(InterceptedService::new(stream, authenticate.clone()))
        .add_service(health)
        .add_service(InterceptedService::new(reflection_v1, authenticate.clone()))
        .add_service(InterceptedService::new(reflection_v1alpha, authenticate)))
}

fn authorize(request: Request<()>, token: Option<&str>) -> Result<Request<()>, Status> {
    let Some(token) = token else {
        return Ok(request);
    };
    let expected = format!("Bearer {token}");
    match request
        .metadata()
        .get("authorization")
        .and_then(|value| value.to_str().ok())
    {
        Some(value) if constant_time_eq(value.as_bytes(), expected.as_bytes()) => Ok(request),
        _ => Err(Status::unauthenticated("invalid bearer token")),
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut different = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        different |= usize::from(
            left.get(index).copied().unwrap_or_default()
                ^ right.get(index).copied().unwrap_or_default(),
        );
    }
    different == 0
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{
        v1::{AckPolicy, SchemaSet, SessionOpen},
        *,
    };
    use prost::Message;
    use tempfile::TempDir;
    use tonic::metadata::MetadataValue;

    #[test]
    fn generated_message_round_trip() {
        let open = SessionOpen {
            client_name: "test-client".into(),
            schema_fingerprint: vec![1, 2, 3],
            schema: Some(SchemaSet::default()),
            ack_policy: Some(AckPolicy {
                max_unacked_rows: 256,
                max_ack_delay_ms: 100,
            }),
            client_instance_id: vec![4, 5, 6],
        };

        let decoded = SessionOpen::decode(open.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded, open);
    }

    #[test]
    fn bearer_comparison_rejects_length_and_value_mismatches() {
        assert!(constant_time_eq(b"Bearer secret", b"Bearer secret"));
        assert!(!constant_time_eq(b"Bearer secret", b"Bearer other!"));
        assert!(!constant_time_eq(b"Bearer secret", b"Bearer secret-long"));
    }

    #[tokio::test]
    async fn serve_listener_accepts_prebound_socket() {
        let directory = TempDir::new().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { serve_listener(listener, db).await.unwrap() });
        let endpoint = format!("http://{addr}");
        let mut query = loop {
            match v1::query_service_client::QueryServiceClient::connect(endpoint.clone()).await {
                Ok(client) => break client,
                Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
            }
        };
        query
            .get_time_range(v1::GetTimeRangeRequest {})
            .await
            .unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn bearer_auth_health_and_reflection() {
        let directory = TempDir::new().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let server = tokio::spawn(async move {
            serve_with_auth(addr, db, Some("secret".into()))
                .await
                .unwrap()
        });
        let endpoint = format!("http://{addr}");
        let channel = loop {
            match tonic::transport::Endpoint::from_shared(endpoint.clone())
                .unwrap()
                .connect()
                .await
            {
                Ok(channel) => break channel,
                Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
            }
        };

        let mut unauthenticated =
            v1::query_service_client::QueryServiceClient::new(channel.clone());
        assert_eq!(
            unauthenticated
                .get_time_range(v1::GetTimeRangeRequest {})
                .await
                .unwrap_err()
                .code(),
            tonic::Code::Unauthenticated
        );

        let bearer = |mut request: Request<()>| {
            request.metadata_mut().insert(
                "authorization",
                MetadataValue::try_from("Bearer secret").unwrap(),
            );
            Ok(request)
        };
        let mut query =
            v1::query_service_client::QueryServiceClient::with_interceptor(channel.clone(), bearer);
        query
            .get_time_range(v1::GetTimeRangeRequest {})
            .await
            .unwrap();

        let mut health = tonic_health::pb::health_client::HealthClient::new(channel.clone());
        let status = health
            .check(tonic_health::pb::HealthCheckRequest {
                service: "elodin.db.v1.QueryService".into(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            status.status,
            tonic_health::pb::health_check_response::ServingStatus::Serving as i32
        );

        use tonic_reflection::pb::v1::{
            ServerReflectionRequest, server_reflection_request, server_reflection_response,
        };
        let mut reflection =
            tonic_reflection::pb::v1::server_reflection_client::ServerReflectionClient::with_interceptor(
                channel,
                bearer,
            );
        let mut responses = reflection
            .server_reflection_info(tokio_stream::iter([ServerReflectionRequest {
                host: String::new(),
                message_request: Some(server_reflection_request::MessageRequest::ListServices(
                    String::new(),
                )),
            }]))
            .await
            .unwrap()
            .into_inner();
        let response = futures_lite::StreamExt::next(&mut responses)
            .await
            .unwrap()
            .unwrap();
        let Some(server_reflection_response::MessageResponse::ListServicesResponse(services)) =
            response.message_response
        else {
            panic!("expected reflection service list");
        };
        assert!(
            services
                .service
                .iter()
                .any(|service| service.name == "elodin.db.v1.QueryService")
        );
        server.abort();
    }
}
