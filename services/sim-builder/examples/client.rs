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

    let channel = if use_vsock {
        let addr = "vsock://3:50051".parse::<Uri>()?;
        Endpoint::from(addr).connect_with_connector_lazy(tower::service_fn(vsock_connect))
    } else {
        let addr = "http://[::1]:50051".parse()?;
        println!("connecting to {addr}");
        Channel::builder(addr).connect_lazy()
    };
    let mut client = BuildSimClient::new(channel)
        .send_compressed(CompressionEncoding::Zstd)
        .accept_compressed(CompressionEncoding::Zstd);

    let request = tonic::Request::new(BuildReq { code });
    let response = client.build(request).await?.into_inner();
    println!("received artifacts ({} bytes)", response.artifacts.len());
    Ok(())
}

async fn vsock_connect(uri: Uri) -> Result<tokio_vsock::VsockStream, std::io::Error> {
    let cid = uri.host().unwrap().parse::<u32>().unwrap();
    let port = uri.port_u16().unwrap();
    let addr = tokio_vsock::VsockAddr::new(cid, port as u32);
    println!("connecting to vsock:{cid}:{port}");
    tokio_vsock::VsockStream::connect(addr).await
}
