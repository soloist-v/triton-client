use http::uri::InvalidUri;
use numpy::{FromVecError, NotContiguousError};
use tonic::Status;
use serde_json::Error as SerdeJsonError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("General error: {0}")]
    Msg(String),
    #[error(transparent)]
    TransportError(#[from] tonic::transport::Error),
    #[error(transparent)]
    InvalidUri(#[from] InvalidUri),
    #[error(transparent)]
    InvalidAccessToken(#[from] tonic::metadata::errors::InvalidMetadataValue),
    #[error(transparent)]
    ResponseError(#[from] Status),
    #[error(transparent)]
    JsonError(#[from] SerdeJsonError),
    #[error(transparent)]
    EncodeError(#[from] prost::EncodeError),
    #[error(transparent)]
    FromVecError(#[from] FromVecError),
    #[error(transparent)]
    NotContiguousError(#[from] NotContiguousError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<anyhow::Error> for Error {
    fn from(value: anyhow::Error) -> Self {
        Self::Msg(format!("{:#}", value))
    }
}

impl From<Error> for pyo3::PyErr {
    fn from(error: Error) -> Self {
        pyo3::exceptions::PyException::new_err(format!("{:#}", error))
    }
}

impl Error {
    pub fn msg(msg: impl ToString) -> Self {
        Self::Msg(msg.to_string())
    }
}
