use std::sync::Arc;

use impeller2::types::ComponentId;
use impeller2_wkt::{ComponentMetadata as DbComponentMetadata, SetDbConfig as DbSetDbConfig};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use super::{
    common,
    v1::{
        self, GetAssetRequest, GetAssetResponse, GetDbConfigRequest, GetDbConfigResponse,
        ListAssetsRequest, ListAssetsResponse, PutAssetRequest, PutAssetResponse,
        SetComponentMetadataRequest, SetComponentMetadataResponse, SetDbConfigRequest,
        SetDbConfigResponse, admin_service_server::AdminService, put_asset_request,
    },
};
use crate::DB;

const ASSET_CHUNK_BYTES: usize = 1024 * 1024;
const MAX_ASSET_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone)]
pub(super) struct AdminServiceImpl {
    db: Arc<DB>,
}

impl AdminServiceImpl {
    pub(super) fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    fn store_asset(&self, key: &str, bytes: &[u8]) -> Result<PutAssetResponse, Status> {
        self.db
            .store_asset_from_client(key, bytes)
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::InvalidInput => Status::invalid_argument(error.to_string()),
                std::io::ErrorKind::PermissionDenied => {
                    Status::permission_denied(error.to_string())
                }
                _ => common::internal(error),
            })?;
        let assets_revision = self
            .db
            .with_state(|state| state.db_config.assets_revision());
        Ok(PutAssetResponse {
            size: bytes.len() as u64,
            assets_revision,
        })
    }
}

#[tonic::async_trait]
impl AdminService for AdminServiceImpl {
    type GetAssetStream = ReceiverStream<Result<GetAssetResponse, Status>>;

    async fn get_db_config(
        &self,
        _request: Request<GetDbConfigRequest>,
    ) -> Result<Response<GetDbConfigResponse>, Status> {
        let config = self
            .db
            .with_state(|state| common::db_config(&state.db_config));
        Ok(Response::new(GetDbConfigResponse {
            config: Some(config),
        }))
    }

    async fn set_db_config(
        &self,
        request: Request<SetDbConfigRequest>,
    ) -> Result<Response<SetDbConfigResponse>, Status> {
        let request = request.into_inner();
        self.db
            .apply_set_db_config_from_client(DbSetDbConfig {
                recording: request.recording,
                metadata: request.metadata,
            })
            .map_err(common::db_error)?;
        let config = self
            .db
            .with_state(|state| common::db_config(&state.db_config));
        Ok(Response::new(SetDbConfigResponse {
            config: Some(config),
        }))
    }

    async fn set_component_metadata(
        &self,
        request: Request<SetComponentMetadataRequest>,
    ) -> Result<Response<SetComponentMetadataResponse>, Status> {
        let metadata = request
            .into_inner()
            .metadata
            .ok_or_else(|| Status::invalid_argument("metadata is required"))?;
        if metadata.name.is_empty() {
            return Err(Status::invalid_argument("component name must be non-empty"));
        }
        let value = DbComponentMetadata {
            component_id: ComponentId::new(&metadata.name),
            name: metadata.name,
            metadata: metadata.metadata,
        };
        self.db
            .with_state_mut(|state| state.set_component_metadata(value.clone(), &self.db.path))
            .map_err(common::db_error)?;
        Ok(Response::new(SetComponentMetadataResponse {
            metadata: Some(common::component_metadata(&value)),
        }))
    }

    async fn put_asset(
        &self,
        request: Request<tonic::Streaming<PutAssetRequest>>,
    ) -> Result<Response<PutAssetResponse>, Status> {
        let mut incoming = request.into_inner();
        let first = incoming
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("asset stream is empty"))?;
        let Some(put_asset_request::Chunk::Header(header)) = first.chunk else {
            return Err(Status::invalid_argument(
                "first asset request must contain a header",
            ));
        };
        if header.key.is_empty() {
            return Err(Status::invalid_argument("asset key must be non-empty"));
        }
        let mut bytes = Vec::new();
        while let Some(request) = incoming.message().await? {
            let Some(put_asset_request::Chunk::Data(data)) = request.chunk else {
                return Err(Status::invalid_argument(
                    "asset header is only allowed once",
                ));
            };
            if bytes.len().saturating_add(data.len()) > MAX_ASSET_BYTES {
                return Err(Status::resource_exhausted("asset exceeds 256 MiB"));
            }
            bytes.extend_from_slice(&data);
        }
        let key = header.key;
        let service = self.clone();
        let response = tokio::task::spawn_blocking(move || service.store_asset(&key, &bytes))
            .await
            .map_err(common::internal)??;
        Ok(Response::new(response))
    }

    async fn get_asset(
        &self,
        request: Request<GetAssetRequest>,
    ) -> Result<Response<Self::GetAssetStream>, Status> {
        let key = request.into_inner().key;
        let path = crate::assets_http::assets_dir(&self.db.path);
        let bytes =
            tokio::task::spawn_blocking(move || crate::assets_http::read_asset_file(&path, &key))
                .await
                .map_err(common::internal)?
                .map_err(|error| match error.kind() {
                    std::io::ErrorKind::NotFound => Status::not_found(error.to_string()),
                    std::io::ErrorKind::InvalidInput => Status::invalid_argument(error.to_string()),
                    _ => common::internal(error),
                })?;
        let (tx, rx) = mpsc::channel(4);
        tokio::spawn(async move {
            for data in bytes.chunks(ASSET_CHUNK_BYTES) {
                if tx
                    .send(Ok(GetAssetResponse {
                        data: data.to_vec(),
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

    async fn list_assets(
        &self,
        request: Request<ListAssetsRequest>,
    ) -> Result<Response<ListAssetsResponse>, Status> {
        let prefix = request.into_inner().prefix;
        let path = crate::assets_http::assets_dir(&self.db.path);
        let assets = tokio::task::spawn_blocking(move || {
            crate::assets::index_assets_in(
                &path,
                if prefix.is_empty() {
                    None
                } else {
                    Some(prefix.as_str())
                },
            )
        })
        .await
        .map_err(common::internal)?
        .map_err(common::internal)?
        .into_iter()
        .map(|entry| v1::AssetInfo {
            key: entry.key,
            size: entry.size,
        })
        .collect();
        Ok(Response::new(ListAssetsResponse { assets }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use tempfile::TempDir;

    use super::*;

    #[tokio::test]
    async fn config_metadata_and_assets_round_trip() {
        let directory = TempDir::new().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let service = AdminServiceImpl::new(db.clone());

        let config = service
            .set_db_config(Request::new(SetDbConfigRequest {
                recording: Some(false),
                metadata: [("demo".into(), "value".into())].into_iter().collect(),
            }))
            .await
            .unwrap()
            .into_inner()
            .config
            .unwrap();
        assert!(!config.recording);
        assert_eq!(config.metadata["demo"], "value");
        let config = service
            .set_db_config(Request::new(SetDbConfigRequest {
                recording: None,
                metadata: [("demo".into(), String::new())].into_iter().collect(),
            }))
            .await
            .unwrap()
            .into_inner()
            .config
            .unwrap();
        assert!(!config.metadata.contains_key("demo"));

        let metadata = service
            .set_component_metadata(Request::new(SetComponentMetadataRequest {
                metadata: Some(v1::ComponentMetadata {
                    name: "demo.signal".into(),
                    metadata: [("unit".into(), "m".into())].into_iter().collect(),
                }),
            }))
            .await
            .unwrap()
            .into_inner()
            .metadata
            .unwrap();
        assert_eq!(metadata.metadata["unit"], "m");

        let stored = service.store_asset("demo/data.bin", b"contents").unwrap();
        assert_eq!(stored.size, 8);
        assert_eq!(stored.assets_revision, 1);
        let mut asset = service
            .get_asset(Request::new(GetAssetRequest {
                key: "demo/data.bin".into(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            futures_lite::StreamExt::next(&mut asset)
                .await
                .unwrap()
                .unwrap()
                .data,
            b"contents"
        );
        let assets = service
            .list_assets(Request::new(ListAssetsRequest {
                prefix: "demo/".into(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(assets.assets[0].key, "demo/data.bin");

        db.assets_read_only.store(true, Ordering::Release);
        assert_eq!(
            service
                .store_asset("demo/rejected.bin", b"x")
                .unwrap_err()
                .code(),
            tonic::Code::PermissionDenied
        );
    }
}
