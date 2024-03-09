use atc_entity::events::Error as EventError;
use elodin_types::ValidationError;
use sea_orm::TransactionError;
use std::io;
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
    #[error("postcard: {0}")]
    Postcard(#[from] postcard::Error),
    #[error("redis: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("kube watcher: {0}")]
    KubeWatcher(#[from] kube::runtime::watcher::Error),
    #[error("sandbox not booted")]
    SandboxNotBooted,
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("axum error: {0}")]
    Axum(#[from] axum::Error),
    #[error("tonic transport error: {0}")]
    TonicTransport(#[from] tonic::transport::Error),
    #[error("signed url error: {0}")]
    SignedURL(#[from] google_cloud_storage::sign::SignedURLError),
}

impl From<EventError> for Error {
    fn from(value: EventError) -> Self {
        match value {
            EventError::Db(err) => Error::Db(err),
            EventError::NotFound => Error::NotFound,
            EventError::Postcard(err) => Error::Postcard(err),
            EventError::Redis(err) => Error::Redis(err),
        }
    }
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
            Error::Postcard(err) => Status::new(Code::Internal, err.to_string()),
            Error::SandboxNotBooted => Status::new(Code::FailedPrecondition, "sandbox not booted"),
            err => Status::new(Code::Internal, err.to_string()),
        }
    }
}

impl From<Error> for Status {
    fn from(val: Error) -> Self {
        val.status()
    }
}

impl axum::response::IntoResponse for Error {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;
        match self {
            Error::Db(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
            Error::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized".to_string()),
            Error::Reqwest(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
            Error::InvalidRequest => (StatusCode::BAD_REQUEST, "invalid request".to_string()),
            Error::Kube(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
            Error::VMBootFailed(err) => (StatusCode::INTERNAL_SERVER_ERROR, err),
            Error::FlumeRecv(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
            Error::FlumeSend => (StatusCode::INTERNAL_SERVER_ERROR, "flume send".to_string()),
            Error::NotFound => (StatusCode::NOT_FOUND, "not found".to_string()),
            Error::Postcard(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
            Error::SandboxNotBooted => (StatusCode::BAD_REQUEST, "sandbox not booted".to_string()),
            err => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
        }
        .into_response()
    }
}
