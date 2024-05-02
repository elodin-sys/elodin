pub mod api {
    use crate::ValidationError;

    tonic::include_proto!("elodin.types.api");

    impl UpdateSandboxReq {
        pub fn id(&self) -> Result<uuid::Uuid, ValidationError> {
            uuid::Uuid::from_slice(&self.id).map_err(|_| ValidationError)
        }
    }

    impl BootSandboxReq {
        pub fn id(&self) -> Result<uuid::Uuid, ValidationError> {
            uuid::Uuid::from_slice(&self.id).map_err(|_| ValidationError)
        }
    }

    impl GetSandboxReq {
        pub fn id(&self) -> Result<uuid::Uuid, ValidationError> {
            uuid::Uuid::from_slice(&self.id).map_err(|_| ValidationError)
        }
    }

    impl DeleteSandboxReq {
        pub fn id(&self) -> Result<uuid::Uuid, ValidationError> {
            uuid::Uuid::from_slice(&self.id).map_err(|_| ValidationError)
        }
    }

    impl Page {
        pub fn last_id(&self) -> Result<uuid::Uuid, ValidationError> {
            uuid::Uuid::from_slice(&self.last_id).map_err(|_| ValidationError)
        }
    }

    impl StartMonteCarloRunReq {
        pub fn id(&self) -> Result<uuid::Uuid, ValidationError> {
            uuid::Uuid::from_slice(&self.id).map_err(|_| ValidationError)
        }
    }

    impl GetMonteCarloRunReq {
        pub fn id(&self) -> Result<uuid::Uuid, ValidationError> {
            uuid::Uuid::from_slice(&self.id).map_err(|_| ValidationError)
        }
    }
}

pub mod sandbox {
    tonic::include_proto!("elodin.types.sandbox");
}

pub use bit_vec::BitVec;

pub struct ValidationError;

pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("elodin_types");

pub const BATCH_TOPIC: &str = "mc:batch";

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Debug, Clone)]
pub struct Batch {
    pub run_id: uuid::Uuid,
    pub batch_no: usize,
    pub buffer: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct Metadata {
    pub entrypoint: String,
}
