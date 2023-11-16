use paracosm_types::api::{
    api_server::{self, ApiServer},
    CreateUserReq, CreateUserResp,
};
use tonic::{async_trait, transport::Server, Response, Status};
use tracing::info;

use crate::config::ApiConfig;

pub struct Api;

impl Api {
    pub async fn run(self, config: ApiConfig) -> anyhow::Result<()> {
        let svc = ApiServer::new(ApiService);
        info!(api.addr = ?config.address, "api listening");
        Server::builder()
            .add_service(svc)
            .serve(config.address)
            .await?;
        Ok(())
    }
}

pub struct ApiService;

#[async_trait]
impl api_server::Api for ApiService {
    async fn create_user(
        &self,
        _req: tonic::Request<CreateUserReq>,
    ) -> Result<Response<CreateUserResp>, Status> {
        Ok(Response::new(CreateUserResp { id: vec![] }))
    }
}
