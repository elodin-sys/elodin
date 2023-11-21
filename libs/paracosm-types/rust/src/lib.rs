use prost::{bytes, Message};

pub mod api {
    use crate::ValidationError;

    tonic::include_proto!("paracosm.types.api");

    impl UpdateSandboxReq {
        pub fn id(&self) -> Result<uuid::Uuid, ValidationError> {
            uuid::Uuid::from_slice(&self.id).map_err(|_| ValidationError)
        }
    }
}

pub struct ValidationError;

pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("paracosm_types");
