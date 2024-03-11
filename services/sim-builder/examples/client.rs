use elodin_types::sandbox::{build_sim_client::BuildSimClient, BuildReq};
use tonic::codec::CompressionEncoding;
use tonic::transport::{Channel, Endpoint, Uri};

const DEFAULT_CODE: &str = include_str!("code.py");

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let use_vsock = std::env::args().any(|arg| arg == "--vsock");
    let code = std::env::var("CODE")
        .map(|p| std::fs::read_to_string(p).unwrap())
        .unwrap_or(DEFAULT_CODE.to_string());

    let addr = if use_vsock {
        "vsock://3:50051"
    } else {
        "http://[::1]:50051"
    };
    let channel = builder_channel(addr.parse().unwrap());
    let mut client = BuildSimClient::new(channel)
        .send_compressed(CompressionEncoding::Zstd)
        .accept_compressed(CompressionEncoding::Zstd);

    let request = tonic::Request::new(BuildReq { code });
    let response = client.build(request).await?.into_inner();
    println!("received artifacts ({} bytes)", response.artifacts.len());
    Ok(())
}

fn builder_channel(addr: Uri) -> Channel {
    let scheme = addr.scheme().map(|s| s.as_str());
    let use_vsock = scheme == Some("vsock");
    let endpoint = Endpoint::from(addr);

    if use_vsock {
        #[cfg(not(target_os = "linux"))]
        panic!("vsock not supported on os");
        #[cfg(target_os = "linux")]
        return endpoint.connect_with_connector_lazy(tower::service_fn(vsock_connect));
    }
    endpoint.connect_lazy()
}

#[cfg(target_os = "linux")]
async fn vsock_connect(uri: Uri) -> Result<tokio_vsock::VsockStream, std::io::Error> {
    let cid = uri.host().unwrap().parse::<u32>().unwrap();
    let port = uri.port_u16().unwrap();
    let addr = tokio_vsock::VsockAddr::new(cid, port as u32);
    println!("connecting to vsock:{cid}:{port}");
    tokio_vsock::VsockStream::connect(addr).await
}
