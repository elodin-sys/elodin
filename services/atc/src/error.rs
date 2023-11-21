use paracosm_types::ValidationError;
use sea_orm::TransactionError;
use thiserror::Error;
use tonic::{Code, Status};
#[derive(Debug, Error)]
pub enum Error {
    #[error("db error: {0}")]
    Db(#[from] sea_orm::DbErr),
    #[error("unauthorized")]
    Unauthorized,
    #[error("upstream http: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("invalid request")]
    InvalidRequest,
}

impl From<TransactionError<Error>> for Error {
    fn from(err: TransactionError<Error>) -> Self {
        match err {
            TransactionError::Connection(db) => Error::Db(db),
            TransactionError::Transaction(e) => e,
        }
    }
}

impl From<ValidationError> for Error {
    fn from(value: ValidationError) -> Self {
        Error::InvalidRequest
    }
}

impl Error {
    pub fn status(self) -> Status {
        match self {
            Error::Db(err) => Status::new(Code::Internal, err.to_string()),
            Error::Unauthorized => Status::new(Code::Unauthenticated, "unauthorized".to_string()),
            Error::Reqwest(err) => Status::new(Code::Internal, err.to_string()),
            Error::InvalidRequest => {
                Status::new(Code::InvalidArgument, "invalid request".to_string())
            }
        }
    }
}

impl From<Error> for Status {
    fn from(val: Error) -> Self {
        val.status()
    }
}
