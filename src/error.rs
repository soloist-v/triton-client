use http::uri::InvalidUri;
use tonic::Status;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("General error: {0}")]
    Msg(String),
    #[error("a gRPC transport error has occurred: {0}")]
    TransportError(#[from] tonic::transport::Error),
    #[error("the client was provided and invalid URI: {0}")]
    InvalidUri(#[from] InvalidUri),
    #[error("invalid access token")]
    InvalidAccessToken(#[from] tonic::metadata::errors::InvalidMetadataValue),
    #[error("grpc call returned error status")]
    ResponseError(#[from] Status),
    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Protobuf encoding error: {0}")]
    EncodeError(#[from] prost::EncodeError),
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
