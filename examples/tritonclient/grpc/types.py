"""Type definitions for tritonclient.grpc."""

import numpy as np
from typing import Optional, List, Dict, Any
import triton_client
from ..utils.dtype import triton_to_np_dtype


class InferInput:
    """Represents an input tensor for inference."""
    
    def __init__(
        self,
        name: str,
        shape: List[int],
        datatype: str,
    ) -> None:
        """Initialize an inference input.
        
        Args:
            name: The name of the input tensor.
            shape: The shape of the input tensor.
            datatype: The data type of the input tensor (e.g., "FP32", "INT8").
        """
        self._name = name
        self._shape = shape
        self._datatype = datatype
        self._data: Optional[np.ndarray] = None
        self._shm_region_name: Optional[str] = None
        self._shm_byte_size: int = 0
        self._shm_offset: int = 0
    
    def name(self) -> str:
        """Get the name of input associated with this object.
        
        Returns:
            The name of input.
        """
        return self._name
    
    def datatype(self) -> str:
        """Get the datatype of input associated with this object.
        
        Returns:
            The datatype of input.
        """
        return self._datatype
    
    def shape(self) -> List[int]:
        """Get the shape of input associated with this object.
        
        Returns:
            The shape of input.
        """
        return self._shape.copy()
    
    def set_shape(self, shape: List[int]) -> "InferInput":
        """Set the shape of the input tensor.
        
        Args:
            shape: The shape of the input tensor.
            
        Returns:
            The updated input.
        """
        self._shape = shape.copy()
        return self
    
    def set_data_from_numpy(self, input_tensor: np.ndarray) -> "InferInput":
        """Set the input data from a numpy array.
        
        Args:
            input_tensor: The numpy array containing the input data.
            
        Returns:
            The updated input.
        """
        self._data = input_tensor
        self._shm_region_name = None  # 清除共享内存设置
        return self
    
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
        self._shm_region_name = region_name
        self._shm_byte_size = byte_size
        self._shm_offset = offset
        self._data = None  # 清除 numpy 数据
        return self
    
    def to_rust_input(self) -> triton_client.InferInputTensor:
        """Convert to Rust InferInputTensor.
        
        Returns:
            A Rust InferInputTensor object.
        """
        contents = None
        parameters: Dict[str, triton_client.InferParameter] = {}
        
        if self._data is not None:
            # 使用 numpy 数据
            contents = self._create_tensor_contents(self._data)
        elif self._shm_region_name is not None:
            # 使用共享内存，contents 为空
            contents = triton_client.InferTensorContents()
            # 设置共享内存参数
            parameters = self._create_shm_parameters()
        else:
            # 没有数据，创建空的 contents
            contents = triton_client.InferTensorContents()
        
        return triton_client.InferInputTensor(
            name=self._name,
            datatype=self._datatype,
            shape=self._shape,
            contents=contents,
            parameters=parameters,
        )
    
    def _create_tensor_contents(self, data: np.ndarray) -> triton_client.InferTensorContents:
        """从 numpy 数组创建 InferTensorContents."""
        contents = triton_client.InferTensorContents()
        dtype = triton_to_np_dtype(self._datatype)
        flat_data = data.flatten()
        
        # 根据数据类型设置内容
        if dtype == np.bool_:
            contents.replace_bool_contents(flat_data)
        elif dtype == np.int8:
            contents.replace_int_contents(flat_data.astype(np.int32))
        elif dtype == np.int16:
            contents.replace_int_contents(flat_data.astype(np.int32))
        elif dtype == np.int32:
            contents.replace_int_contents(flat_data)
        elif dtype == np.int64:
            contents.replace_int64_contents(flat_data)
        elif dtype == np.uint8:
            contents.replace_uint_contents(flat_data.astype(np.uint32))
        elif dtype == np.uint16:
            contents.replace_uint_contents(flat_data.astype(np.uint32))
        elif dtype == np.uint32:
            contents.replace_uint_contents(flat_data)
        elif dtype == np.uint64:
            contents.replace_uint64_contents(flat_data)
        elif dtype == np.float16:
            contents.replace_fp32_contents(flat_data.astype(np.float32))
        elif dtype == np.float32:
            contents.replace_fp32_contents(flat_data)
        elif dtype == np.float64:
            contents.replace_fp64_contents(flat_data)
        else:
            raise ValueError(f"Unsupported data type: {self._datatype}")
        
        return contents
    
    def _create_shm_parameters(self) -> Dict[str, triton_client.InferParameter]:
        """创建共享内存参数字典."""
        parameters = {}
        
        # 设置共享内存区域名称
        shm_name_param = triton_client.InferParameter()
        # ParameterChoice.string_param 是 classmethod，直接调用即可
        shm_name_param.parameter_choice = triton_client.ParameterChoice.string_param(
            self._shm_region_name
        )
        parameters["shared_memory_region"] = shm_name_param
        
        # 设置字节大小
        byte_size_param = triton_client.InferParameter()
        byte_size_param.parameter_choice = triton_client.ParameterChoice.uint64_param(
            self._shm_byte_size
        )
        parameters["shared_memory_byte_size"] = byte_size_param
        
        # 设置偏移量（如果有）
        if self._shm_offset > 0:
            offset_param = triton_client.InferParameter()
            offset_param.parameter_choice = triton_client.ParameterChoice.uint64_param(
                self._shm_offset
            )
            parameters["shared_memory_offset"] = offset_param
        
        return parameters


class InferRequestedOutput:
    """Represents a requested output tensor for inference."""
    
    def __init__(self, name: str, class_count: int = 0) -> None:
        """Initialize a requested output.
        
        Args:
            name: The name of the output tensor.
            class_count: The number of classifications to be requested. Default is 0.
        """
        self._name = name
        self._class_count = class_count
        self._shm_region_name: Optional[str] = None
        self._shm_byte_size: int = 0
        self._shm_offset: int = 0
    
    def name(self) -> str:
        """Get the name of output associated with this object.
        
        Returns:
            The name of output.
        """
        return self._name
    
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
        self._shm_region_name = region_name
        self._shm_byte_size = byte_size
        self._shm_offset = offset
    
    def unset_shared_memory(self) -> None:
        """Clears the shared memory option set by the last call to set_shared_memory().
        After call to this function requested output will no longer be returned in a
        shared memory region.
        """
        self._shm_region_name = None
        self._shm_byte_size = 0
        self._shm_offset = 0
    
    def to_rust_output(self) -> triton_client.InferRequestedOutputTensor:
        """Convert to Rust InferRequestedOutputTensor.
        
        Returns:
            A Rust InferRequestedOutputTensor object.
        """
        parameters: Dict[str, triton_client.InferParameter] = {}
        
        if self._shm_region_name is not None:
            # 设置共享内存参数
            shm_name_param = triton_client.InferParameter()
            # ParameterChoice.string_param 是 classmethod，直接调用即可
            shm_name_param.parameter_choice = triton_client.ParameterChoice.string_param(
                self._shm_region_name
            )
            parameters["shared_memory_region"] = shm_name_param
            
            byte_size_param = triton_client.InferParameter()
            byte_size_param.parameter_choice = triton_client.ParameterChoice.uint64_param(
                self._shm_byte_size
            )
            parameters["shared_memory_byte_size"] = byte_size_param
            
            if self._shm_offset > 0:
                offset_param = triton_client.InferParameter()
                offset_param.parameter_choice = triton_client.ParameterChoice.uint64_param(
                    self._shm_offset
                )
                parameters["shared_memory_offset"] = offset_param
        
        return triton_client.InferRequestedOutputTensor(
            name=self._name,
            parameters=parameters,
        )


class InferOutput:
    """Represents an output tensor from inference."""
    
    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        datatype: str,
        contents: triton_client.InferTensorContents,
    ):
        """Initialize an output tensor.
        
        Args:
            name: The name of the output tensor.
            shape: The shape of the output tensor.
            datatype: The data type of the output tensor.
            contents: The tensor contents from Rust.
        """
        self._name = name
        self._shape = shape
        self._datatype = datatype
        self._contents = contents
    
    @property
    def name(self) -> str:
        """The name of the output tensor."""
        return self._name
    
    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the output tensor."""
        return self._shape
    
    @property
    def datatype(self) -> str:
        """The data type of the output tensor."""
        return self._datatype
    
    def as_numpy(self) -> np.ndarray:
        """Get the output tensor as a numpy array.
        
        Returns:
            The output tensor as a numpy array.
        """
        return self._tensor_contents_to_numpy(self._contents, self._datatype, self._shape)
    
    @staticmethod
    def _tensor_contents_to_numpy(
        contents: triton_client.InferTensorContents,
        datatype: str,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        """将 InferTensorContents 转换为 numpy 数组."""
        import pyo3
        
        dtype = triton_to_np_dtype(datatype)
        py = pyo3.Python.with_gil()
        
        # 创建 contents 的可变副本（因为 replace 方法需要 &mut self）
        # InferTensorContents 实现了 Clone，可以直接克隆
        mut_contents = contents.clone()
        
        # 使用 replace 方法获取数据（所有权转移，零拷贝）
        # replace 方法接受 None 作为参数，只返回当前数据
        if dtype == np.bool_:
            arr = mut_contents.replace_bool_contents(py, None)
        elif dtype == np.int32:
            arr = mut_contents.replace_int_contents(py, None)
        elif dtype == np.int64:
            arr = mut_contents.replace_int64_contents(py, None)
        elif dtype == np.uint32:
            arr = mut_contents.replace_uint_contents(py, None)
        elif dtype == np.uint64:
            arr = mut_contents.replace_uint64_contents(py, None)
        elif dtype == np.float32:
            arr = mut_contents.replace_fp32_contents(py, None)
        elif dtype == np.float64:
            arr = mut_contents.replace_fp64_contents(py, None)
        else:
            # 对于其他类型，尝试使用 int32
            arr = mut_contents.replace_int_contents(py, None)
        
        return arr.reshape(shape)


class InferResult:
    """Represents the result of an inference request."""
    
    def __init__(self, response: triton_client.ModelInferResponse):
        """Initialize an inference result.
        
        Args:
            response: The ModelInferResponse from Rust.
        """
        self._response = response
        self._outputs: Dict[str, InferOutput] = {}
        self._extract_outputs()
    
    def _extract_outputs(self) -> None:
        """提取输出张量."""
        for output in self._response.outputs:
            # 获取形状（get_all 自动生成 getter，shape 是 List[i64]）
            shape_list = output.shape  # 直接访问，get_all 会自动处理
            shape = tuple(int(x) for x in shape_list)  # 转换为 tuple[int, ...]
            
            # 获取数据类型
            datatype = output.datatype
            
            # 获取 contents（可能是 None）
            contents = output.contents if output.contents is not None else triton_client.InferTensorContents()
            
            # 创建 InferOutput
            infer_output = InferOutput(
                name=output.name,
                shape=shape,
                datatype=datatype,
                contents=contents,
            )
            
            self._outputs[output.name] = infer_output
    
    def as_numpy(self, name: str) -> np.ndarray | None:
        """Get the output tensor as a numpy array.
        
        Args:
            name: The name of the output tensor.
            
        Returns:
            The output tensor as a numpy array, or None if not found.
        """
        output = self._outputs.get(name)
        if output is None:
            return None
        return output.as_numpy()
    
    def get_output(self, name: str, as_json: bool = False) -> Any:
        """Get the output tensor object.
        
        Args:
            name: The name of the output tensor.
            as_json: If True then returns response as a json dict, otherwise as a protobuf message.
            
        Returns:
            The output tensor object as a protobuf message or dict, or None if not found.
        """
        output = self._outputs.get(name)
        if output is None:
            return None
        
        if as_json:
            # TODO: 实现 JSON 序列化
            return {
                "name": output.name,
                "shape": list(output.shape),
                "datatype": output.datatype,
            }
        else:
            return output
    
    def get_response(self, as_json: bool = False) -> Any:
        """Retrieves the complete ModelInferResponse as a json dict object or protobuf message.
        
        Args:
            as_json: If True then returns response as a json dict, otherwise as a protobuf message.
            
        Returns:
            The underlying ModelInferResponse as a protobuf message or dict.
        """
        if as_json:
            # TODO: 实现 JSON 序列化
            return {
                "model_name": self._response.model_name,
                "model_version": self._response.model_version,
                "id": self._response.id,
                "outputs": [
                    {
                        "name": output.name,
                        "shape": list(output.shape),
                        "datatype": output.datatype,
                    }
                    for output in self._outputs.values()
                ],
            }
        else:
            return self._response


class TensorMetadata:
    """Represents metadata for a tensor (input or output)."""
    
    def __init__(
        self,
        name: str,
        datatype: str,
        shape: List[int],
    ):
        """Initialize tensor metadata.
        
        Args:
            name: The name of the tensor.
            datatype: The data type of the tensor.
            shape: The shape of the tensor.
        """
        self._name = name
        self._datatype = datatype
        self._shape = shape
    
    @property
    def name(self) -> str:
        """The name of the tensor."""
        return self._name
    
    @property
    def datatype(self) -> str:
        """The data type of the tensor (e.g., "FP32", "INT8")."""
        return self._datatype
    
    @property
    def shape(self) -> List[int]:
        """The shape of the tensor."""
        return self._shape.copy()


class ModelMetadata:
    """Represents model metadata."""
    
    def __init__(self, metadata: triton_client.ModelMetadataResponse):
        """Initialize model metadata.
        
        Args:
            metadata: The ModelMetadataResponse from Rust.
        """
        self._metadata = metadata
        self._inputs = [
            TensorMetadata(
                name=input.name,
                datatype=input.datatype,
                shape=input.shape,
            )
            for input in metadata.inputs
        ]
        self._outputs = [
            TensorMetadata(
                name=output.name,
                datatype=output.datatype,
                shape=output.shape,
            )
            for output in metadata.outputs
        ]
    
    @property
    def name(self) -> str:
        """The name of the model."""
        return self._metadata.name
    
    @property
    def versions(self) -> str:
        """The versions of the model."""
        return ",".join(self._metadata.versions)
    
    @property
    def platform(self) -> str:
        """The platform of the model."""
        return self._metadata.platform
    
    @property
    def inputs(self) -> List[TensorMetadata]:
        """The input specifications of the model."""
        return self._inputs
    
    @property
    def outputs(self) -> List[TensorMetadata]:
        """The output specifications of the model."""
        return self._outputs


class TensorConfig:
    """Represents configuration for a tensor (input or output)."""
    
    def __init__(
        self,
        name: str,
        datatype: str,
        shape: List[int],
    ):
        """Initialize tensor configuration.
        
        Args:
            name: The name of the tensor.
            datatype: The data type of the tensor.
            shape: The shape of the tensor.
        """
        self._name = name
        self._datatype = datatype
        self._shape = shape
    
    @property
    def name(self) -> str:
        """The name of the tensor."""
        return self._name
    
    @property
    def datatype(self) -> str:
        """The data type of the tensor."""
        return self._datatype
    
    @property
    def shape(self) -> List[int]:
        """The shape of the tensor."""
        return self._shape.copy()


class ModelConfig:
    """Represents model configuration."""
    
    def __init__(self, config: triton_client.ModelConfig):
        """Initialize model configuration.
        
        Args:
            config: The ModelConfig from Rust.
        """
        self._config = config
        self._inputs = [
            TensorConfig(
                name=input.name,
                datatype=input.datatype,
                shape=input.shape,
            )
            for input in config.input
        ]
        self._outputs = [
            TensorConfig(
                name=output.name,
                datatype=output.datatype,
                shape=output.shape,
            )
            for output in config.output
        ]
    
    @property
    def name(self) -> str:
        """The name of the model."""
        return self._config.name
    
    @property
    def platform(self) -> str:
        """The platform of the model."""
        return self._config.platform
    
    @property
    def max_batch_size(self) -> int:
        """The maximum batch size for the model."""
        return self._config.max_batch_size
    
    @property
    def input(self) -> List[TensorConfig]:
        """The input specifications of the model."""
        return self._inputs
    
    @property
    def output(self) -> List[TensorConfig]:
        """The output specifications of the model."""
        return self._outputs
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """The parameters of the model. Values have a `.string_value` attribute."""
        # TODO: 实现参数提取
        return {}


class ModelConfigResponse:
    """Response object from get_model_config (protobuf message)."""
    
    def __init__(self, response: triton_client.ModelConfigResponse):
        """Initialize model config response.
        
        Args:
            response: The ModelConfigResponse from Rust.
        """
        self._response = response
        self._config = ModelConfig(response.config) if response.config else None
    
    @property
    def config(self) -> Any:
        """The model configuration (protobuf message)."""
        return self._config


class CallContext:
    """Context for asynchronous inference calls."""
    
    def __init__(self, future):
        """Initialize call context.
        
        Args:
            future: The future object from ThreadPoolExecutor.
        """
        self._future = future
    
    def cancel(self) -> None:
        """Cancel the ongoing inference request."""
        self._future.cancel()

