pub mod api {
    tonic::include_proto!("elodin.types.api");
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

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct SampleMetadata {
    pub run_id: uuid::Uuid,
    pub batch_no: usize,
    pub sample_no: usize,
    pub profile: std::collections::HashMap<String, f64>,
}
