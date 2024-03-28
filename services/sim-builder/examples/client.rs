use std::io::Seek;

use elodin_types::sandbox::FileTransferReq;
use elodin_types::sandbox::{sandbox_client::SandboxClient, *};
use tokio::io::AsyncWriteExt;
use tonic::transport::{Channel, Endpoint, Uri};

const DEFAULT_CODE: &str = include_str!("test_code.py");

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
    let mut client = SandboxClient::new(channel);

    let request = tonic::Request::new(BuildReq { code: code.clone() });
    let response = client.build(request).await?.into_inner();
    println!("response: {:?}", response);
    let mut artifacts = tokio::fs::File::from_std(tempfile::tempfile()?);
    let file_req = FileTransferReq {
        name: response.artifacts_file,
    };
    let mut stream = client.recv_file(file_req).await?.into_inner();
    while let Some(chunk) = stream.message().await? {
        artifacts.write_all(&chunk.data).await?;
    }
    let mut artifacts = artifacts.into_std().await;
    artifacts.rewind()?;

    let buf = std::io::BufReader::new(artifacts);
    let mut tar = tar::Archive::new(buf);
    let tmp_dir = tempfile::tempdir()?;
    tar.unpack(tmp_dir.path())?;
    let artifacts = tmp_dir.path().join("artifacts");

    println!("extracted artifacts to: {:?}", artifacts);
    std::mem::forget(tmp_dir); // don't delete the temp dir

    let test_req = tonic::Request::new(TestReq {
        code: code.clone(),
        ..Default::default()
    });
    let test_res = client.test(test_req).await?.into_inner();
    println!("received test results: {:?}", test_res);

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
