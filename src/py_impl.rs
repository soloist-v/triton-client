use crate::py_types;
use crate::Client;
use anyhow::Context;
use prost::Message;

#[pyo3::pymethods]
impl Client {
    #[new]
    fn py_new(url: &str, token: Option<String>) -> crate::Result<Client> {
        let client = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { Client::new(url, token).await })?;
        Ok(client)
    }

    // Server liveness check - returns boolean
    #[pyo3(name = "server_live")]
    fn py_server_live(&self) -> crate::Result<bool> {
        let response = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.server_live().await })?;
        Ok(response.live)
    }

    // Server readiness check - returns boolean
    #[pyo3(name = "server_ready")]
    fn py_server_ready(&self) -> crate::Result<bool> {
        let response = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.server_ready().await })?;
        Ok(response.ready)
    }

    // Model readiness check - returns boolean
    #[pyo3(name = "model_ready")]
    fn py_model_ready(&self, req: py_types::ModelReadyRequest) -> crate::Result<bool> {
        let response = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.model_ready(req.into()).await })?;
        Ok(response.ready)
    }

    // Server metadata - returns JSON string
    #[pyo3(name = "server_metadata")]
    fn py_server_metadata(&self) -> crate::Result<String> {
        let result = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.server_metadata().await })?;
        
        // Convert protobuf to JSON
        let json = serde_json::to_string(&serde_json::json!({
            "name": result.name,
            "version": result.version,
            "extensions": result.extensions,
        }))?;
        Ok(json)
    }

    // Model metadata - returns JSON string
    #[pyo3(name = "model_metadata")]
    fn py_model_metadata(&self, req: py_types::ModelMetadataRequest) -> crate::Result<String> {
        let result = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.model_metadata(req.into()).await })?;
        
        let json = serde_json::to_string(&serde_json::json!({
            "name": result.name,
            "versions": result.versions,
            "platform": result.platform,
            "inputs": result.inputs.iter().map(|input| {
                serde_json::json!({
                    "name": input.name,
                    "datatype": input.datatype,
                    "shape": input.shape,
                })
            }).collect::<Vec<_>>(),
            "outputs": result.outputs.iter().map(|output| {
                serde_json::json!({
                    "name": output.name,
                    "datatype": output.datatype,
                    "shape": output.shape,
                })
            }).collect::<Vec<_>>(),
        }))?;
        Ok(json)
    }

    // Model infer - returns raw bytes (for now, can be extended)
    #[pyo3(name = "model_infer")]
    fn py_model_infer(&self, req: py_types::ModelInferRequest) -> crate::Result<Vec<u8>> {
        let result = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.model_infer(req.into()).await })?;
        
        // Serialize the entire response as protobuf bytes
        let mut buf = Vec::new();
        result.encode(&mut buf)?;
        Ok(buf)
    }

    // Model config - returns JSON string
    #[pyo3(name = "model_config")]
    fn py_model_config(&self, req: py_types::ModelConfigRequest) -> crate::Result<String> {
        let result = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.model_config(req.into()).await })?;
        
        // Return as JSON - config is a complex nested structure
        let json = serde_json::to_string(&serde_json::json!({
            "config": format!("{:?}", result.config),
        }))?;
        Ok(json)
    }

    // Model statistics - returns JSON string
    #[pyo3(name = "model_statistics")]
    fn py_model_statistics(&self, req: py_types::ModelStatisticsRequest) -> crate::Result<String> {
        let result = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.model_statistics(req.into()).await })?;
        
        let json = serde_json::to_string(&serde_json::json!({
            "model_stats": result.model_stats.iter().map(|stat| {
                serde_json::json!({
                    "name": stat.name,
                    "version": stat.version,
                })
            }).collect::<Vec<_>>(),
        }))?;
        Ok(json)
    }

    // Repository index - returns JSON string
    #[pyo3(name = "repository_index")]
    fn py_repository_index(&self, req: py_types::RepositoryIndexRequest) -> crate::Result<String> {
        let result = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.repository_index(req.into()).await })?;
        
        let json = serde_json::to_string(&serde_json::json!({
            "models": result.models.iter().map(|model| {
                serde_json::json!({
                    "name": model.name,
                    "version": model.version,
                    "state": model.state,
                    "reason": model.reason,
                })
            }).collect::<Vec<_>>(),
        }))?;
        Ok(json)
    }

    // Repository model load - returns success status
    #[pyo3(name = "repository_model_load")]
    fn py_repository_model_load(&self, req: py_types::RepositoryModelLoadRequest) -> crate::Result<bool> {
        crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.repository_model_load(req.into()).await })?;
        Ok(true)
    }

    // Repository model unload - returns success status
    #[pyo3(name = "repository_model_unload")]
    fn py_repository_model_unload(&self, req: py_types::RepositoryModelUnloadRequest) -> crate::Result<bool> {
        crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.repository_model_unload(req.into()).await })?;
        Ok(true)
    }

    // System shared memory status - returns JSON string
    #[pyo3(name = "system_shared_memory_status")]
    fn py_system_shared_memory_status(&self, req: py_types::SystemSharedMemoryStatusRequest) -> crate::Result<String> {
        let result = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.system_shared_memory_status(req.into()).await })?;
        
        let json = serde_json::to_string(&serde_json::json!({
            "regions": result.regions.iter().map(|(name, status)| {
                serde_json::json!({
                    "name": name,
                    "key": status.key,
                    "offset": status.offset,
                    "byte_size": status.byte_size,
                })
            }).collect::<Vec<_>>(),
        }))?;
        Ok(json)
    }

    // System shared memory register - returns success status
    #[pyo3(name = "system_shared_memory_register")]
    fn py_system_shared_memory_register(&self, req: py_types::SystemSharedMemoryRegisterRequest) -> crate::Result<bool> {
        crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.system_shared_memory_register(req.into()).await })?;
        Ok(true)
    }

    // System shared memory unregister - returns success status
    #[pyo3(name = "system_shared_memory_unregister")]
    fn py_system_shared_memory_unregister(&self, req: py_types::SystemSharedMemoryUnregisterRequest) -> crate::Result<bool> {
        crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.system_shared_memory_unregister(req.into()).await })?;
        Ok(true)
    }

    // CUDA shared memory status - returns JSON string
    #[pyo3(name = "cuda_shared_memory_status")]
    fn py_cuda_shared_memory_status(&self, req: py_types::CudaSharedMemoryStatusRequest) -> crate::Result<String> {
        let result = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.cuda_shared_memory_status(req.into()).await })?;
        
        let json = serde_json::to_string(&serde_json::json!({
            "regions": result.regions.iter().map(|(name, status)| {
                serde_json::json!({
                    "name": name,
                    "device_id": status.device_id,
                    "byte_size": status.byte_size,
                })
            }).collect::<Vec<_>>(),
        }))?;
        Ok(json)
    }

    // CUDA shared memory register - returns success status
    #[pyo3(name = "cuda_shared_memory_register")]
    fn py_cuda_shared_memory_register(&self, req: py_types::CudaSharedMemoryRegisterRequest) -> crate::Result<bool> {
        crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.cuda_shared_memory_register(req.into()).await })?;
        Ok(true)
    }

    // CUDA shared memory unregister - returns success status
    #[pyo3(name = "cuda_shared_memory_unregister")]
    fn py_cuda_shared_memory_unregister(&self, req: py_types::CudaSharedMemoryUnregisterRequest) -> crate::Result<bool> {
        crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.cuda_shared_memory_unregister(req.into()).await })?;
        Ok(true)
    }

    // Trace setting - returns JSON string
    #[pyo3(name = "trace_setting")]
    fn py_trace_setting(&self, req: py_types::TraceSettingRequest) -> crate::Result<String> {
        let result = crate::TOKIO_RT
            .get()
            .context("failed to get tokio runtime")?
            .block_on(async { self.trace_setting(req.into()).await })?;
        
        // Convert settings to a simple map for JSON serialization
        let settings_map: std::collections::HashMap<String, String> = result
            .settings
            .into_iter()
            .map(|(k, v)| {
                // Convert SettingValue to string representation
                let value_str = format!("{:?}", v.value);
                (k, value_str)
            })
            .collect();
        
        let json = serde_json::to_string(&serde_json::json!({
            "settings": settings_map,
        }))?;
        Ok(json)
    }
}
