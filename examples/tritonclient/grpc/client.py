"""Triton Inference Server gRPC client implementation."""

import triton_client
from typing import List, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import threading
from .types import (
    InferInput,
    InferRequestedOutput,
    InferResult,
    ModelMetadata,
    ModelConfigResponse,
    CallContext,
)


class InferenceServerClient:
    """Client for communicating with Triton Inference Server via gRPC."""
    
    def __init__(
        self,
        url: str,
        verbose: bool = False,
        ssl: bool = False,
        root_certificates: Optional[str] = None,
        private_key: Optional[str] = None,
        certificate_chain: Optional[str] = None,
        creds: Any = None,
        keepalive_options: Optional[Any] = None,
        channel_args: Optional[List[tuple[str, Any]]] = None,
    ) -> None:
        """Initialize the inference server client.
        
        Args:
            url: The URL of the inference server (e.g., "localhost:8001").
            verbose: Whether to enable verbose logging (ignored).
            ssl: Whether to use SSL/TLS (ignored, handled by access_token).
            root_certificates: Path to root certificates file (ignored).
            private_key: Path to private key file (ignored).
            certificate_chain: Path to certificate chain file (ignored).
            creds: A grpc.ChannelCredentials object (ignored).
            keepalive_options: Object encapsulating various GRPC KeepAlive options (ignored).
            channel_args: List of Tuple pairs ("key", value) to be passed directly to the GRPC channel (ignored).
        """
        # 忽略不支持的参数，保持签名一致性
        self._rust_client = triton_client.Client(url, access_token=None)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._closed = False
    
    def get_model_metadata(
        self,
        model_name: str,
        model_version: str = "",
        headers: Optional[Dict[str, str]] = None,
        as_json: bool = False,
        client_timeout: Optional[float] = None,
    ) -> Any:
        """Get metadata for a model.
        
        Args:
            model_name: The name of the model.
            model_version: The version of the model. Default is empty string.
            headers: Optional dictionary specifying additional headers (ignored).
            as_json: If True then returns model metadata as a json dict, otherwise as a protobuf message.
            client_timeout: Client timeout in seconds (ignored).
            
        Returns:
            The model metadata as a protobuf message or json dict.
        """
        req = triton_client.ModelMetadataRequest(
            name=model_name,
            version=model_version,
        )
        resp = self._rust_client.model_metadata(req)
        
        if as_json:
            # TODO: 实现 JSON 序列化
            return {
                "name": resp.name,
                "versions": resp.versions,
                "platform": resp.platform,
                "inputs": [
                    {
                        "name": inp.name,
                        "datatype": inp.datatype,
                        "shape": inp.shape,
                    }
                    for inp in resp.inputs
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "datatype": out.datatype,
                        "shape": out.shape,
                    }
                    for out in resp.outputs
                ],
            }
        else:
            return ModelMetadata(resp)
    
    def get_model_config(
        self,
        model_name: str,
        model_version: str = "",
        headers: Optional[Dict[str, str]] = None,
        as_json: bool = False,
        client_timeout: Optional[float] = None,
    ) -> "ModelConfigResponse":
        """Get configuration for a model.
        
        Args:
            model_name: The name of the model.
            model_version: The version of the model. Default is empty string.
            headers: Optional dictionary specifying additional headers (ignored).
            as_json: If True then returns configuration as a json dict, otherwise as a protobuf message.
            client_timeout: Client timeout in seconds (ignored).
            
        Returns:
            The ModelConfigResponse protobuf message (has a .config attribute) or json dict.
        """
        req = triton_client.ModelConfigRequest(
            name=model_name,
            version=model_version,
        )
        resp = self._rust_client.model_config(req)
        
        if as_json:
            # TODO: 实现 JSON 序列化
            return {
                "config": {
                    "name": resp.config.name,
                    "platform": resp.config.platform,
                    "max_batch_size": resp.config.max_batch_size,
                    "input": [
                        {
                            "name": inp.name,
                            "datatype": inp.datatype,
                            "shape": inp.shape,
                        }
                        for inp in resp.config.input
                    ],
                    "output": [
                        {
                            "name": out.name,
                            "datatype": out.datatype,
                            "shape": out.shape,
                        }
                        for out in resp.config.output
                    ],
                }
            }
        else:
            return ModelConfigResponse(resp)
    
    def get_server_metadata(
        self,
        headers: Optional[Dict[str, str]] = None,
        as_json: bool = False,
        client_timeout: Optional[float] = None,
    ) -> Any:
        """Get metadata for the inference server.
        
        Args:
            headers: Optional dictionary specifying additional headers (ignored).
            as_json: If True then returns server metadata as a json dict, otherwise as a protobuf message.
            client_timeout: Client timeout in seconds (ignored).
            
        Returns:
            The server metadata as a protobuf message or json dict.
        """
        resp = self._rust_client.server_metadata()
        
        if as_json:
            # TODO: 实现 JSON 序列化
            return {
                "name": resp.name,
                "version": resp.version,
                "extensions": resp.extensions,
            }
        else:
            return resp
    
    def register_system_shared_memory(
        self,
        name: str,
        key: str,
        byte_size: int,
        offset: int = 0,
        headers: Optional[Dict[str, str]] = None,
        client_timeout: Optional[float] = None,
    ) -> None:
        """Register a system shared memory region with the server.
        
        Args:
            name: The name of the shared memory region.
            key: The key/identifier of the shared memory region.
            byte_size: The size of the shared memory region in bytes.
            offset: Offset, in bytes, within the underlying memory object. Default is 0.
            headers: Optional dictionary specifying additional headers (ignored).
            client_timeout: Client timeout in seconds (ignored).
        """
        req = triton_client.SystemSharedMemoryRegisterRequest(
            name=name,
            key=key,
            byte_size=byte_size,
            offset=offset,
        )
        self._rust_client.system_shared_memory_register(req)
    
    def unregister_system_shared_memory(
        self,
        name: str = "",
        headers: Optional[Dict[str, str]] = None,
        client_timeout: Optional[float] = None,
    ) -> None:
        """Unregister a system shared memory region from the server.
        
        Args:
            name: The name of the shared memory region to unregister. Default is empty string
                  which means all the system shared memory regions will be unregistered.
            headers: Optional dictionary specifying additional headers (ignored).
            client_timeout: Client timeout in seconds (ignored).
        """
        req = triton_client.SystemSharedMemoryUnregisterRequest(
            name=name,
        )
        self._rust_client.system_shared_memory_unregister(req)
    
    def infer(
        self,
        model_name: str,
        inputs: List[InferInput],
        model_version: str = "",
        outputs: Optional[List[InferRequestedOutput]] = None,
        request_id: str = "",
        sequence_id: int = 0,
        sequence_start: bool = False,
        sequence_end: bool = False,
        priority: int = 0,
        timeout: Optional[float] = None,
        client_timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        compression_algorithm: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> InferResult:
        """Perform synchronous inference.
        
        Args:
            model_name: The name of the model.
            inputs: List of input tensors.
            outputs: List of requested output tensors (optional).
            model_version: The version of the model (optional).
            request_id: Unique identifier for the request (ignored).
            sequence_id: Sequence identifier for stateful models (ignored).
            sequence_start: Whether this is the start of a sequence (ignored).
            sequence_end: Whether this is the end of a sequence (ignored).
            priority: Request priority (ignored).
            timeout: Request timeout in seconds (ignored).
            client_timeout: Client timeout in seconds (ignored).
            headers: Additional headers (ignored).
            compression_algorithm: Compression algorithm to use (ignored).
            parameters: Additional parameters (ignored).
            
        Returns:
            The inference result.
        """
        # 转换输入
        input_tensors = [inp.to_rust_input() for inp in inputs]
        
        # 转换输出
        output_tensors = []
        if outputs:
            output_tensors = [out.to_rust_output() for out in outputs]
        
        # 创建请求
        request = triton_client.ModelInferRequest(
            model_name=model_name,
            model_version=model_version,
            id=request_id,
            inputs=input_tensors,
            outputs=output_tensors,
            parameters={},  # TODO: 转换 parameters
        )
        
        # 执行推理（通过 extract 实现所有权转移，零拷贝）
        response = self._rust_client.model_infer(request)
        
        return InferResult(response)
    
    def async_infer(
        self,
        model_name: str,
        inputs: List[InferInput],
        callback: Callable[[InferResult, Optional[Exception]], None],
        model_version: str = "",
        outputs: Optional[List[InferRequestedOutput]] = None,
        request_id: str = "",
        sequence_id: int = 0,
        sequence_start: bool = False,
        sequence_end: bool = False,
        priority: int = 0,
        timeout: Optional[float] = None,
        client_timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        compression_algorithm: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> CallContext:
        """Perform asynchronous inference.
        
        Args:
            model_name: The name of the model.
            inputs: List of input tensors.
            callback: Callback function called when inference completes.
                      The callback receives (result, error) as arguments.
            model_version: The version of the model (optional).
            outputs: List of requested output tensors (optional).
            request_id: Unique identifier for the request (ignored).
            sequence_id: Sequence identifier for stateful models (ignored).
            sequence_start: Whether this is the start of a sequence (ignored).
            sequence_end: Whether this is the end of a sequence (ignored).
            priority: Request priority (ignored).
            timeout: Request timeout in seconds (ignored).
            client_timeout: Client timeout in seconds (ignored).
            headers: Additional headers (ignored).
            compression_algorithm: Compression algorithm to use (ignored).
            parameters: Additional parameters (ignored).
            
        Returns:
            A call context that can be used to cancel the request.
        """
        def worker():
            try:
                result = self.infer(
                    model_name=model_name,
                    inputs=inputs,
                    model_version=model_version,
                    outputs=outputs,
                    request_id=request_id,
                    sequence_id=sequence_id,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    priority=priority,
                    timeout=timeout,
                    client_timeout=client_timeout,
                    headers=headers,
                    compression_algorithm=compression_algorithm,
                    parameters=parameters,
                )
                callback(result, None)
            except Exception as e:
                callback(None, e)
        
        future = self._executor.submit(worker)
        return CallContext(future)
    
    def close(self) -> None:
        """Close the client connection."""
        if not self._closed:
            self._executor.shutdown(wait=True)
            self._closed = True

