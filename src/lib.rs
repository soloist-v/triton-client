#![doc = include_str!("../README.md")]

pub mod client;
mod error;
mod inference;
mod py_types;
mod utils;

pub use client::Client;
pub use error::{Error, Result};

use pyo3::prelude::*;
use tokio::sync::OnceCell;

static TOKIO_RT: OnceCell<tokio::runtime::Runtime> = OnceCell::const_new();

#[pymodule(gil_used = false)]
fn triton_client(m: &Bound<'_, PyModule>) -> PyResult<()> {
    utils::init_log(std::env::var("LOG_LEVEL").unwrap_or("INFO".to_string()));
    let rt = tokio::runtime::Runtime::new()?;
    if let Err(e) = TOKIO_RT.set(rt) {
        log::error!("TRITON_CLIENT RUNTIME SET ERROR: {:#}", e);
    }
    // Add client class
    m.add_class::<Client>()?;
    // Add request/response types
    m.add_class::<inference::ServerLiveResponse>()?;
    m.add_class::<inference::ServerReadyResponse>()?;
    m.add_class::<inference::ModelReadyRequest>()?;
    m.add_class::<inference::ModelReadyResponse>()?;
    m.add_class::<inference::ServerMetadataResponse>()?;
    m.add_class::<inference::ModelMetadataRequest>()?;
    m.add_class::<inference::ModelMetadataResponse>()?;
    m.add_class::<inference::ModelInferRequest>()?;
    m.add_class::<inference::ModelInferResponse>()?;
    m.add_class::<inference::ModelConfigRequest>()?;
    m.add_class::<inference::ModelConfigResponse>()?;
    m.add_class::<inference::ModelStatisticsRequest>()?;
    m.add_class::<inference::ModelStatisticsResponse>()?;
    m.add_class::<inference::RepositoryIndexRequest>()?;
    m.add_class::<inference::RepositoryIndexResponse>()?;
    m.add_class::<inference::RepositoryModelLoadRequest>()?;
    m.add_class::<inference::RepositoryModelLoadResponse>()?;
    m.add_class::<inference::RepositoryModelUnloadRequest>()?;
    m.add_class::<inference::RepositoryModelUnloadResponse>()?;
    m.add_class::<inference::SystemSharedMemoryStatusRequest>()?;
    m.add_class::<inference::SystemSharedMemoryStatusResponse>()?;
    m.add_class::<inference::SystemSharedMemoryRegisterRequest>()?;
    m.add_class::<inference::SystemSharedMemoryRegisterResponse>()?;
    m.add_class::<inference::SystemSharedMemoryUnregisterRequest>()?;
    m.add_class::<inference::SystemSharedMemoryUnregisterResponse>()?;
    m.add_class::<inference::CudaSharedMemoryStatusRequest>()?;
    m.add_class::<inference::CudaSharedMemoryStatusResponse>()?;
    m.add_class::<inference::CudaSharedMemoryRegisterRequest>()?;
    m.add_class::<inference::CudaSharedMemoryRegisterResponse>()?;
    m.add_class::<inference::CudaSharedMemoryUnregisterRequest>()?;
    m.add_class::<inference::CudaSharedMemoryUnregisterResponse>()?;
    m.add_class::<inference::TraceSettingRequest>()?;
    m.add_class::<inference::TraceSettingResponse>()?;
    // child types
    m.add_class::<inference::system_shared_memory_status_response::RegionStatus>()?;
    m.add_class::<inference::ModelRepositoryParameter>()?;
    m.add_class::<inference::model_repository_parameter::ParameterChoice>()?;
    m.add_class::<inference::model_metadata_response::TensorMetadata>()?;
    m.add_class::<inference::InferParameter>()?;
    m.add_class::<inference::infer_parameter::ParameterChoice>()?;
    m.add_class::<inference::model_infer_request::InferInputTensor>()?;
    m.add_class::<inference::model_infer_request::InferRequestedOutputTensor>()?;
    m.add_class::<inference::InferTensorContents>()?;
    m.add_class::<inference::ModelConfig>()?;
    m.add_class::<inference::model_infer_response::InferOutputTensor>()?;
    m.add_class::<inference::repository_index_response::ModelIndex>()?;
    Ok(())
}
