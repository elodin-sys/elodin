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
    #[error("kube error: {0}")]
    Kube(#[from] kube::Error),
    #[error("vm boot error: {0}")]
    VMBootFailed(String),
    #[error("recv error: {0}")]
    FlumeRecv(#[from] flume::RecvError),
    #[error("send error")]
    FlumeSend,
    #[error("not found")]
    NotFound,
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
    fn from(_: ValidationError) -> Self {
        Error::InvalidRequest
    }
}

impl<T> From<flume::SendError<T>> for Error {
    fn from(_: flume::SendError<T>) -> Self {
        Error::FlumeSend
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
            Error::Kube(err) => Status::new(Code::Internal, err.to_string()),
            Error::VMBootFailed(err) => Status::new(Code::Internal, err),
            Error::FlumeRecv(err) => Status::new(Code::Internal, err.to_string()),
            Error::FlumeSend => Status::new(Code::Internal, "flume send".to_string()),
            Error::NotFound => Status::new(Code::NotFound, "not found".to_string()),
        }
    }
}

impl From<Error> for Status {
    fn from(val: Error) -> Self {
        val.status()
    }
}
