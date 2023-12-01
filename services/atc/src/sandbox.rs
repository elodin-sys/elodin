use paracosm_types::sandbox::{sandbox_control_client::SandboxControlClient, UpdateCodeReq};
use crate::error::Error;
use tonic::transport::Channel;

pub async fn update_sandbox_code(vm_ip: &str, code: String) -> Result<(), Error> {
    let Ok(ip) = format!("grpc://{}:50051", vm_ip).parse() else {
        return Err(Error::VMBootFailed("vm has invalid ip".to_string()));
    };
    let channel = Channel::builder(ip).connect().await?;
    let mut client = SandboxControlClient::new(channel);
    let res = client
        .update_code(UpdateCodeReq { code })
        .await
        .map_err(|err| Error::VMBootFailed(err.to_string()))?;
    let _res = res.into_inner();
    Ok(())
}
