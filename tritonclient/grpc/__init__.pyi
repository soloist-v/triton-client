"""Type stubs for tritonclient.grpc module."""

from typing import Any, Callable, Optional
import numpy as np

__all__ = [
    "InferenceServerClient",
    "InferInput",
    "InferRequestedOutput",
    "InferResult",
    "InferOutput",
    "ModelMetadata",
    "TensorMetadata",
    "ModelConfig",
    "ModelConfigResponse",
    "TensorConfig",
    "CallContext",
]


class InferInput:
    """Represents an input tensor for inference."""
    
    def __init__(
        self,
        name: str,
        shape: list[int],
        datatype: str,
    ) -> None:
        """Initialize an inference input.
        
        Args:
            name: The name of the input tensor.
            shape: The shape of the input tensor.
            datatype: The data type of the input tensor (e.g., "FP32", "INT8").
        """
        ...
    
    def name(self) -> str:
        """Get the name of input associated with this object.
        
        Returns:
            The name of input.
        """
        ...
    
    def datatype(self) -> str:
        """Get the datatype of input associated with this object.
        
        Returns:
            The datatype of input.
        """
        ...
    
    def shape(self) -> list[int]:
        """Get the shape of input associated with this object.
        
        Returns:
            The shape of input.
        """
        ...
    
    def set_shape(self, shape: list[int]) -> "InferInput":
        """Set the shape of the input tensor.
        
        Args:
            shape: The shape of the input tensor.
            
        Returns:
            The updated input.
        """
        ...
    
    def set_data_from_numpy(self, input_tensor: np.ndarray) -> "InferInput":
        """Set the input data from a numpy array.
        
        Args:
            input_tensor: The numpy array containing the input data.
            
        Returns:
            The updated input.
        """
        ...
    
    def set_shared_memory(
        self,
        region_name: str,
        byte_size: int,
        offset: int = 0,
    ) -> "InferInput":
        """Set the input to use shared memory.
        
        Args:
            region_name: The name of the shared memory region.
            byte_size: The size of the data in bytes.
            offset: The offset, in bytes, into the region where the data starts. Default is 0.
            
        Returns:
            The updated input.
        """
        ...


class InferRequestedOutput:
    """Represents a requested output tensor for inference."""
    
    def __init__(self, name: str, class_count: int = 0) -> None:
        """Initialize a requested output.
        
        Args:
            name: The name of the output tensor.
            class_count: The number of classifications to be requested. Default is 0.
        """
        ...
    
    def name(self) -> str:
        """Get the name of output associated with this object.
        
        Returns:
            The name of output.
        """
        ...
    
    def set_shared_memory(
        self,
        region_name: str,
        byte_size: int,
        offset: int = 0,
    ) -> None:
        """Set the output to use shared memory.
        
        Args:
            region_name: The name of the shared memory region.
            byte_size: The size of the data in bytes.
            offset: The offset, in bytes, into the region where the data starts. Default is 0.
        """
        ...
    
    def unset_shared_memory(self) -> None:
        """Clears the shared memory option set by the last call to set_shared_memory().
        After call to this function requested output will no longer be returned in a
        shared memory region.
        """
        ...


class InferOutput:
    """Represents an output tensor from inference."""
    
    @property
    def name(self) -> str:
        """The name of the output tensor."""
        ...
    
    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the output tensor."""
        ...
    
    @property
    def datatype(self) -> str:
        """The data type of the output tensor."""
        ...


class InferResult:
    """Represents the result of an inference request."""
    
    def as_numpy(self, name: str) -> np.ndarray | None:
        """Get the output tensor as a numpy array.
        
        Args:
            name: The name of the output tensor.
            
        Returns:
            The output tensor as a numpy array, or None if not found.
        """
        ...
    
    def get_output(self, name: str, as_json: bool = False) -> Any:
        """Get the output tensor object.
        
        Args:
            name: The name of the output tensor.
            as_json: If True then returns response as a json dict, otherwise as a protobuf message.
            
        Returns:
            The output tensor object as a protobuf message or dict, or None if not found.
        """
        ...
    
    def get_response(self, as_json: bool = False) -> Any:
        """Retrieves the complete ModelInferResponse as a json dict object or protobuf message.
        
        Args:
            as_json: If True then returns response as a json dict, otherwise as a protobuf message.
            
        Returns:
            The underlying ModelInferResponse as a protobuf message or dict.
        """
        ...


class TensorMetadata:
    """Represents metadata for a tensor (input or output)."""
    
    @property
    def name(self) -> str:
        """The name of the tensor."""
        ...
    
    @property
    def datatype(self) -> str:
        """The data type of the tensor (e.g., "FP32", "INT8")."""
        ...
    
    @property
    def shape(self) -> list[int]:
        """The shape of the tensor."""
        ...


class ModelMetadata:
    """Represents model metadata."""
    
    @property
    def name(self) -> str:
        """The name of the model."""
        ...
    
    @property
    def versions(self) -> str:
        """The versions of the model."""
        ...
    
    @property
    def platform(self) -> str:
        """The platform of the model."""
        ...
    
    @property
    def inputs(self) -> list[TensorMetadata]:
        """The input specifications of the model."""
        ...
    
    @property
    def outputs(self) -> list[TensorMetadata]:
        """The output specifications of the model."""
        ...


class TensorConfig:
    """Represents configuration for a tensor (input or output)."""
    
    @property
    def name(self) -> str:
        """The name of the tensor."""
        ...
    
    @property
    def datatype(self) -> str:
        """The data type of the tensor."""
        ...
    
    @property
    def shape(self) -> list[int]:
        """The shape of the tensor."""
        ...


class ModelConfig:
    """Represents model configuration."""
    
    @property
    def name(self) -> str:
        """The name of the model."""
        ...
    
    @property
    def platform(self) -> str:
        """The platform of the model."""
        ...
    
    @property
    def max_batch_size(self) -> int:
        """The maximum batch size for the model."""
        ...
    
    @property
    def input(self) -> list[TensorConfig]:
        """The input specifications of the model."""
        ...
    
    @property
    def output(self) -> list[TensorConfig]:
        """The output specifications of the model."""
        ...
    
    @property
    def parameters(self) -> dict[str, Any]:
        """The parameters of the model. Values have a `.string_value` attribute."""
        ...


class CallContext:
    """Context for asynchronous inference calls."""
    
    def cancel(self) -> None:
        """Cancel the ongoing inference request."""
        ...


class ModelConfigResponse:
    """Response object from get_model_config (protobuf message)."""
    
    @property
    def config(self) -> Any:
        """The model configuration (protobuf message)."""
        ...


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
        channel_args: Optional[list[tuple[str, Any]]] = None,
    ) -> None:
        """Initialize the inference server client.
        
        Args:
            url: The URL of the inference server (e.g., "localhost:8001").
            verbose: Whether to enable verbose logging.
            ssl: Whether to use SSL/TLS.
            root_certificates: Path to root certificates file.
            private_key: Path to private key file.
            certificate_chain: Path to certificate chain file.
            creds: A grpc.ChannelCredentials object to use for the connection.
            keepalive_options: Object encapsulating various GRPC KeepAlive options.
            channel_args: List of Tuple pairs ("key", value) to be passed directly to the GRPC channel.
        """
        ...
    
    def get_model_metadata(
        self,
        model_name: str,
        model_version: str = "",
        headers: Optional[dict[str, str]] = None,
        as_json: bool = False,
        client_timeout: Optional[float] = None,
    ) -> Any:
        """Get metadata for a model.
        
        Args:
            model_name: The name of the model.
            model_version: The version of the model. Default is empty string.
            headers: Optional dictionary specifying additional headers.
            as_json: If True then returns model metadata as a json dict, otherwise as a protobuf message.
            client_timeout: Client timeout in seconds (optional).
            
        Returns:
            The model metadata as a protobuf message or json dict.
        """
        ...
    
    def get_model_config(
        self,
        model_name: str,
        model_version: str = "",
        headers: Optional[dict[str, str]] = None,
        as_json: bool = False,
        client_timeout: Optional[float] = None,
    ) -> "ModelConfigResponse":
        """Get configuration for a model.
        
        Args:
            model_name: The name of the model.
            model_version: The version of the model. Default is empty string.
            headers: Optional dictionary specifying additional headers.
            as_json: If True then returns configuration as a json dict, otherwise as a protobuf message.
            client_timeout: Client timeout in seconds (optional).
            
        Returns:
            The ModelConfigResponse protobuf message (has a .config attribute) or json dict.
        """
        ...
    
    def get_server_metadata(
        self,
        headers: Optional[dict[str, str]] = None,
        as_json: bool = False,
        client_timeout: Optional[float] = None,
    ) -> Any:
        """Get metadata for the inference server.
        
        Args:
            headers: Optional dictionary specifying additional headers.
            as_json: If True then returns server metadata as a json dict, otherwise as a protobuf message.
            client_timeout: Client timeout in seconds (optional).
            
        Returns:
            The server metadata as a protobuf message or json dict.
        """
        ...
    
    def register_system_shared_memory(
        self,
        name: str,
        key: str,
        byte_size: int,
        offset: int = 0,
        headers: Optional[dict[str, str]] = None,
        client_timeout: Optional[float] = None,
    ) -> None:
        """Register a system shared memory region with the server.
        
        Args:
            name: The name of the shared memory region.
            key: The key/identifier of the shared memory region.
            byte_size: The size of the shared memory region in bytes.
            offset: Offset, in bytes, within the underlying memory object. Default is 0.
            headers: Optional dictionary specifying additional headers.
            client_timeout: Client timeout in seconds (optional).
        """
        ...
    
    def unregister_system_shared_memory(
        self,
        name: str = "",
        headers: Optional[dict[str, str]] = None,
        client_timeout: Optional[float] = None,
    ) -> None:
        """Unregister a system shared memory region from the server.
        
        Args:
            name: The name of the shared memory region to unregister. Default is empty string
                  which means all the system shared memory regions will be unregistered.
            headers: Optional dictionary specifying additional headers.
            client_timeout: Client timeout in seconds (optional).
        """
        ...
    
    def infer(
        self,
        model_name: str,
        inputs: list[InferInput],
        model_version: str = "",
        outputs: Optional[list[InferRequestedOutput]] = None,
        request_id: str = "",
        sequence_id: int = 0,
        sequence_start: bool = False,
        sequence_end: bool = False,
        priority: int = 0,
        timeout: Optional[float] = None,
        client_timeout: Optional[float] = None,
        headers: Optional[dict[str, str]] = None,
        compression_algorithm: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
    ) -> InferResult:
        """Perform synchronous inference.
        
        Args:
            model_name: The name of the model.
            inputs: List of input tensors.
            outputs: List of requested output tensors (optional).
            model_version: The version of the model (optional).
            request_id: Unique identifier for the request (optional).
            sequence_id: Sequence identifier for stateful models (optional).
            sequence_start: Whether this is the start of a sequence (optional).
            sequence_end: Whether this is the end of a sequence (optional).
            priority: Request priority (optional).
            timeout: Request timeout in seconds (optional).
            client_timeout: Client timeout in seconds (optional).
            headers: Additional headers (optional).
            compression_algorithm: Compression algorithm to use (optional).
            
        Returns:
            The inference result.
        """
        ...
    
    def async_infer(
        self,
        model_name: str,
        inputs: list[InferInput],
        callback: Callable[[InferResult, Optional[Exception]], None],
        model_version: str = "",
        outputs: Optional[list[InferRequestedOutput]] = None,
        request_id: str = "",
        sequence_id: int = 0,
        sequence_start: bool = False,
        sequence_end: bool = False,
        priority: int = 0,
        timeout: Optional[float] = None,
        client_timeout: Optional[float] = None,
        headers: Optional[dict[str, str]] = None,
        compression_algorithm: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
    ) -> CallContext:
        """Perform asynchronous inference.
        
        Args:
            model_name: The name of the model.
            inputs: List of input tensors.
            outputs: List of requested output tensors (optional).
            model_version: The version of the model (optional).
            request_id: Unique identifier for the request (optional).
            sequence_id: Sequence identifier for stateful models (optional).
            sequence_start: Whether this is the start of a sequence (optional).
            sequence_end: Whether this is the end of a sequence (optional).
            priority: Request priority (optional).
            timeout: Request timeout in seconds (optional).
            client_timeout: Client timeout in seconds (optional).
            headers: Additional headers (optional).
            compression_algorithm: Compression algorithm to use (optional).
            callback: Callback function called when inference completes (optional).
                      The callback receives (result, error) as arguments.
            
        Returns:
            A call context that can be used to cancel the request.
        """
        ...
    
    def close(self) -> None:
        """Close the client connection."""
        ...

