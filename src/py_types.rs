use pyo3::prelude::*;
use std::collections::HashMap;

/// Python wrapper for ModelReadyRequest
#[pyclass]
#[derive(Clone)]
pub struct ModelReadyRequest {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub version: String,
}

#[pymethods]
impl ModelReadyRequest {
    #[new]
    #[pyo3(signature = (name, version="".to_string()))]
    fn new(name: String, version: String) -> Self {
        Self { name, version }
    }
}

impl From<ModelReadyRequest> for crate::inference::ModelReadyRequest {
    fn from(req: ModelReadyRequest) -> Self {
        Self {
            name: req.name,
            version: req.version,
        }
    }
}

/// Python wrapper for ModelMetadataRequest
#[pyclass]
#[derive(Clone)]
pub struct ModelMetadataRequest {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub version: String,
}

#[pymethods]
impl ModelMetadataRequest {
    #[new]
    #[pyo3(signature = (name, version="".to_string()))]
    fn new(name: String, version: String) -> Self {
        Self { name, version }
    }
}

impl From<ModelMetadataRequest> for crate::inference::ModelMetadataRequest {
    fn from(req: ModelMetadataRequest) -> Self {
        Self {
            name: req.name,
            version: req.version,
        }
    }
}

/// Python wrapper for ModelConfigRequest
#[pyclass]
#[derive(Clone)]
pub struct ModelConfigRequest {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub version: String,
}

#[pymethods]
impl ModelConfigRequest {
    #[new]
    #[pyo3(signature = (name, version="".to_string()))]
    fn new(name: String, version: String) -> Self {
        Self { name, version }
    }
}

impl From<ModelConfigRequest> for crate::inference::ModelConfigRequest {
    fn from(req: ModelConfigRequest) -> Self {
        Self {
            name: req.name,
            version: req.version,
        }
    }
}

/// Python wrapper for ModelStatisticsRequest
#[pyclass]
#[derive(Clone)]
pub struct ModelStatisticsRequest {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub version: String,
}

#[pymethods]
impl ModelStatisticsRequest {
    #[new]
    #[pyo3(signature = (name, version="".to_string()))]
    fn new(name: String, version: String) -> Self {
        Self { name, version }
    }
}

impl From<ModelStatisticsRequest> for crate::inference::ModelStatisticsRequest {
    fn from(req: ModelStatisticsRequest) -> Self {
        Self {
            name: req.name,
            version: req.version,
        }
    }
}

/// Python wrapper for RepositoryIndexRequest
#[pyclass]
#[derive(Clone, Default)]
pub struct RepositoryIndexRequest {
    #[pyo3(get, set)]
    pub repository_name: String,
    #[pyo3(get, set)]
    pub ready: bool,
}

#[pymethods]
impl RepositoryIndexRequest {
    #[new]
    #[pyo3(signature = (repository_name="".to_string(), ready=false))]
    fn new(repository_name: String, ready: bool) -> Self {
        Self {
            repository_name,
            ready,
        }
    }
}

impl From<RepositoryIndexRequest> for crate::inference::RepositoryIndexRequest {
    fn from(req: RepositoryIndexRequest) -> Self {
        Self {
            repository_name: req.repository_name,
            ready: req.ready,
        }
    }
}

/// Python wrapper for RepositoryModelLoadRequest
#[pyclass]
#[derive(Clone)]
pub struct RepositoryModelLoadRequest {
    #[pyo3(get, set)]
    pub repository_name: String,
    #[pyo3(get, set)]
    pub model_name: String,
}

#[pymethods]
impl RepositoryModelLoadRequest {
    #[new]
    #[pyo3(signature = (model_name, repository_name="".to_string()))]
    fn new(model_name: String, repository_name: String) -> Self {
        Self {
            repository_name,
            model_name,
        }
    }
}

impl From<RepositoryModelLoadRequest> for crate::inference::RepositoryModelLoadRequest {
    fn from(req: RepositoryModelLoadRequest) -> Self {
        Self {
            repository_name: req.repository_name,
            model_name: req.model_name,
            parameters: HashMap::new(),
        }
    }
}

/// Python wrapper for RepositoryModelUnloadRequest
#[pyclass]
#[derive(Clone)]
pub struct RepositoryModelUnloadRequest {
    #[pyo3(get, set)]
    pub repository_name: String,
    #[pyo3(get, set)]
    pub model_name: String,
}

#[pymethods]
impl RepositoryModelUnloadRequest {
    #[new]
    #[pyo3(signature = (model_name, repository_name="".to_string()))]
    fn new(model_name: String, repository_name: String) -> Self {
        Self {
            repository_name,
            model_name,
        }
    }
}

impl From<RepositoryModelUnloadRequest> for crate::inference::RepositoryModelUnloadRequest {
    fn from(req: RepositoryModelUnloadRequest) -> Self {
        Self {
            repository_name: req.repository_name,
            model_name: req.model_name,
            parameters: HashMap::new(),
        }
    }
}

/// Python wrapper for SystemSharedMemoryStatusRequest
#[pyclass]
#[derive(Clone, Default)]
pub struct SystemSharedMemoryStatusRequest {
    #[pyo3(get, set)]
    pub name: String,
}

#[pymethods]
impl SystemSharedMemoryStatusRequest {
    #[new]
    #[pyo3(signature = (name="".to_string()))]
    fn new(name: String) -> Self {
        Self { name }
    }
}

impl From<SystemSharedMemoryStatusRequest> for crate::inference::SystemSharedMemoryStatusRequest {
    fn from(req: SystemSharedMemoryStatusRequest) -> Self {
        Self { name: req.name }
    }
}

/// Python wrapper for SystemSharedMemoryRegisterRequest
#[pyclass]
#[derive(Clone)]
pub struct SystemSharedMemoryRegisterRequest {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub key: String,
    #[pyo3(get, set)]
    pub offset: u64,
    #[pyo3(get, set)]
    pub byte_size: u64,
}

#[pymethods]
impl SystemSharedMemoryRegisterRequest {
    #[new]
    fn new(name: String, key: String, offset: u64, byte_size: u64) -> Self {
        Self {
            name,
            key,
            offset,
            byte_size,
        }
    }
}

impl From<SystemSharedMemoryRegisterRequest>
    for crate::inference::SystemSharedMemoryRegisterRequest
{
    fn from(req: SystemSharedMemoryRegisterRequest) -> Self {
        Self {
            name: req.name,
            key: req.key,
            offset: req.offset,
            byte_size: req.byte_size,
        }
    }
}

/// Python wrapper for SystemSharedMemoryUnregisterRequest
#[pyclass]
#[derive(Clone, Default)]
pub struct SystemSharedMemoryUnregisterRequest {
    #[pyo3(get, set)]
    pub name: String,
}

#[pymethods]
impl SystemSharedMemoryUnregisterRequest {
    #[new]
    #[pyo3(signature = (name="".to_string()))]
    fn new(name: String) -> Self {
        Self { name }
    }
}

impl From<SystemSharedMemoryUnregisterRequest>
    for crate::inference::SystemSharedMemoryUnregisterRequest
{
    fn from(req: SystemSharedMemoryUnregisterRequest) -> Self {
        Self { name: req.name }
    }
}

/// Python wrapper for CudaSharedMemoryStatusRequest
#[pyclass]
#[derive(Clone, Default)]
pub struct CudaSharedMemoryStatusRequest {
    #[pyo3(get, set)]
    pub name: String,
}

#[pymethods]
impl CudaSharedMemoryStatusRequest {
    #[new]
    #[pyo3(signature = (name="".to_string()))]
    fn new(name: String) -> Self {
        Self { name }
    }
}

impl From<CudaSharedMemoryStatusRequest> for crate::inference::CudaSharedMemoryStatusRequest {
    fn from(req: CudaSharedMemoryStatusRequest) -> Self {
        Self { name: req.name }
    }
}

/// Python wrapper for CudaSharedMemoryRegisterRequest
#[pyclass]
#[derive(Clone)]
pub struct CudaSharedMemoryRegisterRequest {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub raw_handle: Vec<u8>,
    #[pyo3(get, set)]
    pub device_id: i64,
    #[pyo3(get, set)]
    pub byte_size: u64,
}

#[pymethods]
impl CudaSharedMemoryRegisterRequest {
    #[new]
    fn new(name: String, raw_handle: Vec<u8>, device_id: i64, byte_size: u64) -> Self {
        Self {
            name,
            raw_handle,
            device_id,
            byte_size,
        }
    }
}

impl From<CudaSharedMemoryRegisterRequest> for crate::inference::CudaSharedMemoryRegisterRequest {
    fn from(req: CudaSharedMemoryRegisterRequest) -> Self {
        Self {
            name: req.name,
            raw_handle: req.raw_handle,
            device_id: req.device_id,
            byte_size: req.byte_size,
        }
    }
}

/// Python wrapper for CudaSharedMemoryUnregisterRequest
#[pyclass]
#[derive(Clone, Default)]
pub struct CudaSharedMemoryUnregisterRequest {
    #[pyo3(get, set)]
    pub name: String,
}

#[pymethods]
impl CudaSharedMemoryUnregisterRequest {
    #[new]
    #[pyo3(signature = (name="".to_string()))]
    fn new(name: String) -> Self {
        Self { name }
    }
}

impl From<CudaSharedMemoryUnregisterRequest>
    for crate::inference::CudaSharedMemoryUnregisterRequest
{
    fn from(req: CudaSharedMemoryUnregisterRequest) -> Self {
        Self { name: req.name }
    }
}

/// Python wrapper for TraceSettingRequest
#[pyclass]
#[derive(Clone, Default)]
pub struct TraceSettingRequest {
    #[pyo3(get, set)]
    pub model_name: String,
}

#[pymethods]
impl TraceSettingRequest {
    #[new]
    #[pyo3(signature = (model_name="".to_string()))]
    fn new(model_name: String) -> Self {
        Self { model_name }
    }
}

impl From<TraceSettingRequest> for crate::inference::TraceSettingRequest {
    fn from(req: TraceSettingRequest) -> Self {
        Self {
            model_name: req.model_name,
            settings: HashMap::new(),
        }
    }
}

/// Python wrapper for InferInput
#[pyclass]
#[derive(Clone)]
pub struct InferInput {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub datatype: String,
    #[pyo3(get, set)]
    pub shape: Vec<i64>,
}

#[pymethods]
impl InferInput {
    #[new]
    fn new(name: String, datatype: String, shape: Vec<i64>) -> Self {
        Self {
            name,
            datatype,
            shape,
        }
    }
}

/// Python wrapper for InferRequestedOutputTensor
#[pyclass]
#[derive(Clone)]
pub struct InferRequestedOutputTensor {
    #[pyo3(get, set)]
    pub name: String,
}

#[pymethods]
impl InferRequestedOutputTensor {
    #[new]
    fn new(name: String) -> Self {
        Self { name }
    }
}

impl From<InferRequestedOutputTensor>
    for crate::inference::model_infer_request::InferRequestedOutputTensor
{
    fn from(req: InferRequestedOutputTensor) -> Self {
        Self {
            name: req.name,
            parameters: HashMap::new(),
        }
    }
}

/// Python wrapper for ModelInferRequest
#[pyclass]
#[derive(Clone)]
pub struct ModelInferRequest {
    #[pyo3(get, set)]
    pub model_name: String,
    #[pyo3(get, set)]
    pub model_version: String,
    #[pyo3(get, set)]
    pub id: String,
    pub inputs: Vec<InferInput>,
    pub outputs: Vec<InferRequestedOutputTensor>,
    pub raw_input_contents: Vec<Vec<u8>>,
}

#[pymethods]
impl ModelInferRequest {
    #[new]
    #[pyo3(signature = (model_name, model_version="".to_string(), id="".to_string()))]
    fn new(model_name: String, model_version: String, id: String) -> Self {
        Self {
            model_name,
            model_version,
            id,
            inputs: Vec::new(),
            outputs: Vec::new(),
            raw_input_contents: Vec::new(),
        }
    }

    fn add_input(&mut self, input: InferInput) {
        self.inputs.push(input);
    }

    fn add_output(&mut self, output: InferRequestedOutputTensor) {
        self.outputs.push(output);
    }

    fn add_raw_input(&mut self, data: Vec<u8>) {
        self.raw_input_contents.push(data);
    }
}

impl From<ModelInferRequest> for crate::inference::ModelInferRequest {
    fn from(req: ModelInferRequest) -> Self {
        Self {
            model_name: req.model_name,
            model_version: req.model_version,
            id: req.id,
            parameters: HashMap::new(),
            inputs: req
                .inputs
                .into_iter()
                .map(|input| crate::inference::model_infer_request::InferInputTensor {
                    name: input.name,
                    datatype: input.datatype,
                    shape: input.shape,
                    parameters: HashMap::new(),
                    contents: None,
                })
                .collect(),
            outputs: req.outputs.into_iter().map(|o| o.into()).collect(),
            raw_input_contents: req.raw_input_contents,
        }
    }
}

