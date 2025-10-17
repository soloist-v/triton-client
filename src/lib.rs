#![doc = include_str!("../README.md")]

pub mod client;
mod error;
mod inference;
mod py_impl;
mod py_types;
pub use client::Client;
pub use error::{Error, Result};
pub use inference::*;

use pyo3::prelude::*;
use tokio::sync::OnceCell;

static TOKIO_RT: OnceCell<tokio::runtime::Runtime> = OnceCell::const_new();
fn init_log(level: String) {
    fn detailed_format(
        w: &mut dyn std::io::Write,
        now: &mut flexi_logger::DeferredNow,
        record: &log::Record,
    ) -> Result<(), std::io::Error> {
        let ts = now.format("%Y-%m-%d %H:%M:%S%.3f");
        let level = match record.level() {
            log::Level::Error => "ERROR",
            log::Level::Warn => "WARNING",
            log::Level::Info => "INFO",
            log::Level::Debug => "DEBUG",
            log::Level::Trace => "TRACE",
        };
        let module = record
            .module_path()
            .unwrap_or("<unnamed>")
            .replace("::", ".");
        write!(
            w,
            "{} | {:<8} | {}:{} - {}",
            ts,
            level,
            module,
            record.line().unwrap_or(0),
            record.args()
        )
    }
    flexi_logger::Logger::try_with_str(level)
        .unwrap()
        .format(detailed_format)
        .log_to_stdout()
        .start()
        .unwrap();
}

#[pymodule(gil_used = false)]
fn triton_client(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_log(std::env::var("LOG_LEVEL").unwrap_or("INFO".to_string()));
    let rt = tokio::runtime::Runtime::new()?;
    TOKIO_RT.set(rt).expect("async runtime already initialized");
    
    // Add client class
    m.add_class::<Client>()?;
    
    // Add request/response types
    m.add_class::<py_types::ModelReadyRequest>()?;
    m.add_class::<py_types::ModelMetadataRequest>()?;
    m.add_class::<py_types::ModelConfigRequest>()?;
    m.add_class::<py_types::ModelStatisticsRequest>()?;
    m.add_class::<py_types::RepositoryIndexRequest>()?;
    m.add_class::<py_types::RepositoryModelLoadRequest>()?;
    m.add_class::<py_types::RepositoryModelUnloadRequest>()?;
    m.add_class::<py_types::SystemSharedMemoryStatusRequest>()?;
    m.add_class::<py_types::SystemSharedMemoryRegisterRequest>()?;
    m.add_class::<py_types::SystemSharedMemoryUnregisterRequest>()?;
    m.add_class::<py_types::CudaSharedMemoryStatusRequest>()?;
    m.add_class::<py_types::CudaSharedMemoryRegisterRequest>()?;
    m.add_class::<py_types::CudaSharedMemoryUnregisterRequest>()?;
    m.add_class::<py_types::TraceSettingRequest>()?;
    m.add_class::<py_types::InferInput>()?;
    m.add_class::<py_types::InferRequestedOutputTensor>()?;
    m.add_class::<py_types::ModelInferRequest>()?;
    
    Ok(())
}
