use std::io;

use impeller2::types::{ComponentId, EntityId};
use thiserror::Error;
#[derive(Debug, Error)]
pub enum Error {
    #[error("map overflow")]
    MapOverflow,
    #[error("stellerator {0}")]
    Stellar(stellarator::Error),
    #[error("io {0}")]
    Io(#[from] std::io::Error),
    #[error("impeller_stella {0}")]
    ImpellerStella(impeller2_stella::Error),
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
    #[error("invalid asset id")]
    InvalidAssetId,
}

impl From<impeller2_stella::Error> for Error {
    fn from(value: impeller2_stella::Error) -> Self {
        match value {
            impeller2_stella::Error::Stellerator(error) => Error::from(error),
            err => Error::ImpellerStella(err),
        }
    }
}

impl From<stellarator::Error> for Error {
    fn from(value: stellarator::Error) -> Self {
        match value {
            stellarator::Error::Io(err) => Error::from(err),
            err => Error::Stellar(err),
        }
    }
}

impl Error {
    pub fn is_stream_closed(&self) -> bool {
        match self {
            Error::Stellar(stellarator::Error::EOF) => true,
            Error::Io(err)
                if err.kind() == io::ErrorKind::BrokenPipe
                    || err.kind() == io::ErrorKind::ConnectionReset =>
            {
                true
            }
            _ => false,
        }
    }
}
