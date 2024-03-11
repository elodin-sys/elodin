use tokio::task::JoinSet;
use tonic::codec::CompressionEncoding;
use tonic::transport::{server::Routes, Server};

mod api;

const MAX_CODE_SIZE: usize = 64 * 1024; // 64 KiB
const MAX_ARTIFACTS_SIZE: usize = 256 * 1024; // 256 KiB

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let mut tasks = JoinSet::new();

    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    api::Service::set_serving(&mut health_reporter).await;

    let build_service = api::BuildSimServer::new(api::Service::default())
        .send_compressed(CompressionEncoding::Zstd)
        .accept_compressed(CompressionEncoding::Zstd)
        .max_decoding_message_size(MAX_CODE_SIZE)
        .max_encoding_message_size(MAX_ARTIFACTS_SIZE);

    let routes = Routes::default()
        .add_service(health_service)
        .add_service(build_service);

    let mut server = Server::builder().trace_fn(|_| tracing::info_span!("grpc"));

    match tokio::net::TcpListener::bind("[::1]:50051").await {
        Ok(tcp_listener) => {
            let tcp_addr = tcp_listener.local_addr()?;
            tracing::info!(?tcp_addr, "listening");
            let tcp_stream = tokio_stream::wrappers::TcpListenerStream::new(tcp_listener);
            let router = server.add_routes(routes.clone());
            tasks.spawn(async move {
                if let Err(err) = router.serve_with_incoming(tcp_stream).await {
                    tracing::error!(?err, "tcp stream terminated");
                }
            });
        }
        Err(err) => tracing::warn!(?err, "failed to bind tcp socket"),
    }

    #[cfg(target_os = "linux")]
    {
        let vsock_addr = tokio_vsock::VsockAddr::new(tokio_vsock::VMADDR_CID_ANY, 50051);
        match tokio_vsock::VsockListener::bind(vsock_addr) {
            Ok(vsock_listener) => {
                tracing::info!(?vsock_addr, "listening");
                let vsock_stream = vsock_listener.incoming();
                let router = server.add_routes(routes);
                tasks.spawn(async move {
                    if let Err(err) = router.serve_with_incoming(vsock_stream).await {
                        tracing::error!(?err, "vsock stream terminated");
                    }
                });
            }
            Err(err) => tracing::warn!(?err, "failed to bind vsock"),
        }
    }

    while let Some(res) = tasks.join_next().await {
        res.unwrap();
    }

    Ok(())
}
