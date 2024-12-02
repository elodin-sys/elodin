use impeller2::types::{ComponentId, EntityId};
use thiserror::Error;
#[derive(Debug, Error)]
pub enum Error {
    #[error("map overflow")]
    MapOverflow,
    #[error("stellerator {0}")]
    Stella(#[from] stellarator::Error),
    #[error("io {0}")]
    Io(#[from] std::io::Error),
    #[error("impeller_stella {0}")]
    ImpellerStella(#[from] impeller2_stella::Error),
    #[error("impeller {0}")]
    Impeller(#[from] impeller2::error::Error),
    #[error("component not found {0}")]
    ComponentNotFound(ComponentId),
    #[error("entity not found {0}")]
    EntityNotFound(EntityId),
    #[error("postcard error {0}")]
    Postcard(#[from] postcard::Error),
    #[error("invalid component id")]
    InvalidComponentId,
}
