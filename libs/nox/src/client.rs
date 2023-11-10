pub struct Client(pub(crate) xla::PjRtClient);

impl Client {
    pub fn cpu() -> Result<Self, xla::Error> {
        xla::PjRtClient::cpu().map(Client)
    }

    pub fn gpu() -> Result<Self, xla::Error> {
        xla::PjRtClient::gpu(0.95, false).map(Client)
    }
}
