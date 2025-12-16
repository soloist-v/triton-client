"""
High-performance Triton inference client
"""
from __future__ import annotations
from . import triton_client
__all__: list = ['__doc__', 'Client', 'ServerLiveResponse', 'ServerReadyResponse', 'ModelReadyRequest', 'ModelReadyResponse', 'ServerMetadataResponse', 'ModelConfig', 'ModelMetadataRequest', 'ModelMetadataResponse', 'ModelInferRequest', 'ModelInferResponse', 'ModelConfigRequest', 'ModelConfigResponse', 'ModelStatisticsRequest', 'ModelStatisticsResponse', 'TraceSettingRequest', 'TraceSettingResponse', 'InferParameter', 'InferTensorContents', 'ModelRepositoryParameter', 'RepositoryIndexRequest', 'RepositoryIndexResponse', 'RepositoryModelLoadRequest', 'RepositoryModelLoadResponse', 'RepositoryModelUnloadRequest', 'RepositoryModelUnloadResponse', 'SystemSharedMemoryStatusRequest', 'SystemSharedMemoryStatusResponse', 'SystemSharedMemoryRegisterRequest', 'SystemSharedMemoryRegisterResponse', 'SystemSharedMemoryUnregisterRequest', 'SystemSharedMemoryUnregisterResponse', 'CudaSharedMemoryStatusRequest', 'CudaSharedMemoryStatusResponse', 'CudaSharedMemoryRegisterRequest', 'CudaSharedMemoryRegisterResponse', 'CudaSharedMemoryUnregisterRequest', 'CudaSharedMemoryUnregisterResponse', 'ParameterChoice', 'TensorMetadata', 'ParameterChoice', 'InferInputTensor', 'RegionStatus', 'InferRequestedOutputTensor', 'InferOutputTensor', 'ModelIndex', 'ListBool', 'ListI8', 'ListI16', 'ListI32', 'ListI64', 'ListU8', 'ListU16', 'ListU32', 'ListU64', 'ListF32', 'ListF64']
class Client:
    """
    Triton Client
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def cuda_shared_memory_register(self, req):
        """
        Register a CUDA-shared-memory region.
        """
    def cuda_shared_memory_status(self, req):
        """
        Get the status of all registered CUDA-shared-memory regions.
        """
    def cuda_shared_memory_unregister(self, req):
        """
        Unregister a CUDA-shared-memory region.
        """
    def model_config(self, req):
        """
        Get model configuration.
        """
    def model_infer(self, req):
        """
        Perform inference using a specific model.
        """
    def model_metadata(self, req):
        """
        Get model metadata.
        """
    def model_ready(self, req):
        """
        Check readiness of a model in the inference server.
        """
    def model_statistics(self, req):
        """
        Get the cumulative inference statistics for a model.
        """
    def repository_index(self, req):
        """
        Get the index of model repository contents.
        """
    def repository_model_load(self, req):
        """
        Load or reload a model from a repository.
        """
    def repository_model_unload(self, req):
        """
        Unload a model.
        """
    def server_live(self):
        """
        Check liveness of the inference server.
        """
    def server_metadata(self):
        """
        Get server metadata.
        """
    def server_ready(self):
        """
        Check readiness of the inference server.
        """
    def system_shared_memory_register(self, req):
        """
        Register a system-shared-memory region.
        """
    def system_shared_memory_status(self, req):
        """
        Get the status of all registered system-shared-memory regions.
        """
    def system_shared_memory_unregister(self, req):
        """
        Unregister a system-shared-memory region.
        """
    def trace_setting(self, req):
        """
        Update and get the trace setting of the Triton server.
        """
class CudaSharedMemoryRegisterRequest:
    """
    @@
    @@.. cpp:var:: message CudaSharedMemoryRegisterRequest
    @@
    @@   Request message for CudaSharedMemoryRegister.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def Get_raw_handle(self):
        ...
    def Replace_raw_handle(self, val):
        ...
    def Set_raw_handle(self, list):
        ...
    def Take_raw_handle(self):
        ...
class CudaSharedMemoryRegisterResponse:
    """
    @@
    @@.. cpp:var:: message CudaSharedMemoryRegisterResponse
    @@
    @@   Response message for CudaSharedMemoryRegister.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class CudaSharedMemoryStatusRequest:
    """
    @@
    @@.. cpp:var:: message CudaSharedMemoryStatusRequest
    @@
    @@   Request message for CudaSharedMemoryStatus.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class CudaSharedMemoryStatusResponse:
    """
    @@
    @@.. cpp:var:: message CudaSharedMemoryStatusResponse
    @@
    @@   Response message for CudaSharedMemoryStatus.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class CudaSharedMemoryUnregisterRequest:
    """
    @@
    @@.. cpp:var:: message CudaSharedMemoryUnregisterRequest
    @@
    @@   Request message for CudaSharedMemoryUnregister.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class CudaSharedMemoryUnregisterResponse:
    """
    @@
    @@.. cpp:var:: message CudaSharedMemoryUnregisterResponse
    @@
    @@   Response message for CudaSharedMemoryUnregister.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class InferInputTensor:
    """
    @@
    @@  .. cpp:var:: message InferInputTensor
    @@
    @@     An input tensor for an inference request.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def Get_shape(self):
        ...
    def Replace_shape(self, val):
        ...
    def Set_shape(self, list):
        ...
    def Take_shape(self):
        ...
class InferOutputTensor:
    """
    @@
    @@  .. cpp:var:: message InferOutputTensor
    @@
    @@     An output tensor returned for an inference request.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def Get_shape(self):
        ...
    def Replace_shape(self, val):
        ...
    def Set_shape(self, list):
        ...
    def Take_shape(self):
        ...
class InferParameter:
    """
    @@
    @@.. cpp:var:: message InferParameter
    @@
    @@   An inference parameter value.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class InferRequestedOutputTensor:
    """
    @@
    @@  .. cpp:var:: message InferRequestedOutputTensor
    @@
    @@     An output tensor requested for an inference request.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class InferTensorContents:
    """
    @@
    @@.. cpp:var:: message InferTensorContents
    @@
    @@   The data contained in a tensor represented by the repeated type
    @@   that matches the tensor's data type. Protobuf oneof is not used
    @@   because oneofs cannot contain repeated fields.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def Get_bool_contents(self):
        ...
    def Get_fp32_contents(self):
        ...
    def Get_fp64_contents(self):
        ...
    def Get_int64_contents(self):
        ...
    def Get_int_contents(self):
        ...
    def Get_uint64_contents(self):
        ...
    def Get_uint_contents(self):
        ...
    def Replace_bool_contents(self, val):
        ...
    def Replace_fp32_contents(self, val):
        ...
    def Replace_fp64_contents(self, val):
        ...
    def Replace_int64_contents(self, val):
        ...
    def Replace_int_contents(self, val):
        ...
    def Replace_uint64_contents(self, val):
        ...
    def Replace_uint_contents(self, val):
        ...
    def Set_bool_contents(self, list):
        ...
    def Set_fp32_contents(self, list):
        ...
    def Set_fp64_contents(self, list):
        ...
    def Set_int64_contents(self, list):
        ...
    def Set_int_contents(self, list):
        ...
    def Set_uint64_contents(self, list):
        ...
    def Set_uint_contents(self, list):
        ...
    def Take_bool_contents(self):
        ...
    def Take_fp32_contents(self):
        ...
    def Take_fp64_contents(self):
        ...
    def Take_int64_contents(self):
        ...
    def Take_int_contents(self):
        ...
    def Take_uint64_contents(self):
        ...
    def Take_uint_contents(self):
        ...
class ListBool:
    """
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_array(arr):
        """
        from numpy array construct
        """
    def __delitem__(self, key):
        """
        Delete self[key].
        """
    def __getitem__(self, key):
        """
        Return self[key].
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __setitem__(self, key, value):
        """
        Set self[key] to value.
        """
    def append(self, item):
        """
        push item
        """
    def clear(self):
        """
        clear
        """
    def copy(self):
        """
        copy
        """
    def insert(self, index, value):
        """
        insert item
        """
    def into_array(self):
        """
        move to numpy array
        """
    def remove(self, index):
        """
        remove item
        """
    def to_array(self):
        """
        to numpy array
        """
    def to_list(self):
        """
        to Python list
        """
class ListF32:
    """
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_array(arr):
        """
        from numpy array construct
        """
    def __delitem__(self, key):
        """
        Delete self[key].
        """
    def __getitem__(self, key):
        """
        Return self[key].
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __setitem__(self, key, value):
        """
        Set self[key] to value.
        """
    def append(self, item):
        """
        push item
        """
    def clear(self):
        """
        clear
        """
    def copy(self):
        """
        copy
        """
    def insert(self, index, value):
        """
        insert item
        """
    def into_array(self):
        """
        move to numpy array
        """
    def remove(self, index):
        """
        remove item
        """
    def to_array(self):
        """
        to numpy array
        """
    def to_list(self):
        """
        to Python list
        """
class ListF64:
    """
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_array(arr):
        """
        from numpy array construct
        """
    def __delitem__(self, key):
        """
        Delete self[key].
        """
    def __getitem__(self, key):
        """
        Return self[key].
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __setitem__(self, key, value):
        """
        Set self[key] to value.
        """
    def append(self, item):
        """
        push item
        """
    def clear(self):
        """
        clear
        """
    def copy(self):
        """
        copy
        """
    def insert(self, index, value):
        """
        insert item
        """
    def into_array(self):
        """
        move to numpy array
        """
    def remove(self, index):
        """
        remove item
        """
    def to_array(self):
        """
        to numpy array
        """
    def to_list(self):
        """
        to Python list
        """
class ListI16:
    """
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_array(arr):
        """
        from numpy array construct
        """
    def __delitem__(self, key):
        """
        Delete self[key].
        """
    def __getitem__(self, key):
        """
        Return self[key].
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __setitem__(self, key, value):
        """
        Set self[key] to value.
        """
    def append(self, item):
        """
        push item
        """
    def clear(self):
        """
        clear
        """
    def copy(self):
        """
        copy
        """
    def insert(self, index, value):
        """
        insert item
        """
    def into_array(self):
        """
        move to numpy array
        """
    def remove(self, index):
        """
        remove item
        """
    def to_array(self):
        """
        to numpy array
        """
    def to_list(self):
        """
        to Python list
        """
class ListI32:
    """
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_array(arr):
        """
        from numpy array construct
        """
    def __delitem__(self, key):
        """
        Delete self[key].
        """
    def __getitem__(self, key):
        """
        Return self[key].
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __setitem__(self, key, value):
        """
        Set self[key] to value.
        """
    def append(self, item):
        """
        push item
        """
    def clear(self):
        """
        clear
        """
    def copy(self):
        """
        copy
        """
    def insert(self, index, value):
        """
        insert item
        """
    def into_array(self):
        """
        move to numpy array
        """
    def remove(self, index):
        """
        remove item
        """
    def to_array(self):
        """
        to numpy array
        """
    def to_list(self):
        """
        to Python list
        """
class ListI64:
    """
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_array(arr):
        """
        from numpy array construct
        """
    def __delitem__(self, key):
        """
        Delete self[key].
        """
    def __getitem__(self, key):
        """
        Return self[key].
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __setitem__(self, key, value):
        """
        Set self[key] to value.
        """
    def append(self, item):
        """
        push item
        """
    def clear(self):
        """
        clear
        """
    def copy(self):
        """
        copy
        """
    def insert(self, index, value):
        """
        insert item
        """
    def into_array(self):
        """
        move to numpy array
        """
    def remove(self, index):
        """
        remove item
        """
    def to_array(self):
        """
        to numpy array
        """
    def to_list(self):
        """
        to Python list
        """
class ListI8:
    """
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_array(arr):
        """
        from numpy array construct
        """
    def __delitem__(self, key):
        """
        Delete self[key].
        """
    def __getitem__(self, key):
        """
        Return self[key].
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __setitem__(self, key, value):
        """
        Set self[key] to value.
        """
    def append(self, item):
        """
        push item
        """
    def clear(self):
        """
        clear
        """
    def copy(self):
        """
        copy
        """
    def insert(self, index, value):
        """
        insert item
        """
    def into_array(self):
        """
        move to numpy array
        """
    def remove(self, index):
        """
        remove item
        """
    def to_array(self):
        """
        to numpy array
        """
    def to_list(self):
        """
        to Python list
        """
class ListU16:
    """
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_array(arr):
        """
        from numpy array construct
        """
    def __delitem__(self, key):
        """
        Delete self[key].
        """
    def __getitem__(self, key):
        """
        Return self[key].
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __setitem__(self, key, value):
        """
        Set self[key] to value.
        """
    def append(self, item):
        """
        push item
        """
    def clear(self):
        """
        clear
        """
    def copy(self):
        """
        copy
        """
    def insert(self, index, value):
        """
        insert item
        """
    def into_array(self):
        """
        move to numpy array
        """
    def remove(self, index):
        """
        remove item
        """
    def to_array(self):
        """
        to numpy array
        """
    def to_list(self):
        """
        to Python list
        """
class ListU32:
    """
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_array(arr):
        """
        from numpy array construct
        """
    def __delitem__(self, key):
        """
        Delete self[key].
        """
    def __getitem__(self, key):
        """
        Return self[key].
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __setitem__(self, key, value):
        """
        Set self[key] to value.
        """
    def append(self, item):
        """
        push item
        """
    def clear(self):
        """
        clear
        """
    def copy(self):
        """
        copy
        """
    def insert(self, index, value):
        """
        insert item
        """
    def into_array(self):
        """
        move to numpy array
        """
    def remove(self, index):
        """
        remove item
        """
    def to_array(self):
        """
        to numpy array
        """
    def to_list(self):
        """
        to Python list
        """
class ListU64:
    """
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_array(arr):
        """
        from numpy array construct
        """
    def __delitem__(self, key):
        """
        Delete self[key].
        """
    def __getitem__(self, key):
        """
        Return self[key].
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __setitem__(self, key, value):
        """
        Set self[key] to value.
        """
    def append(self, item):
        """
        push item
        """
    def clear(self):
        """
        clear
        """
    def copy(self):
        """
        copy
        """
    def insert(self, index, value):
        """
        insert item
        """
    def into_array(self):
        """
        move to numpy array
        """
    def remove(self, index):
        """
        remove item
        """
    def to_array(self):
        """
        to numpy array
        """
    def to_list(self):
        """
        to Python list
        """
class ListU8:
    """
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_array(arr):
        """
        from numpy array construct
        """
    def __delitem__(self, key):
        """
        Delete self[key].
        """
    def __getitem__(self, key):
        """
        Return self[key].
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __setitem__(self, key, value):
        """
        Set self[key] to value.
        """
    def append(self, item):
        """
        push item
        """
    def clear(self):
        """
        clear
        """
    def copy(self):
        """
        copy
        """
    def insert(self, index, value):
        """
        insert item
        """
    def into_array(self):
        """
        move to numpy array
        """
    def remove(self, index):
        """
        remove item
        """
    def to_array(self):
        """
        to numpy array
        """
    def to_list(self):
        """
        to Python list
        """
class ModelConfig:
    """
    @@
    @@.. cpp:var:: message ModelConfig
    @@
    @@   A model configuration.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ModelConfigRequest:
    """
    @@
    @@.. cpp:var:: message ModelConfigRequest
    @@
    @@   Request message for ModelConfig.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ModelConfigResponse:
    """
    @@
    @@.. cpp:var:: message ModelConfigResponse
    @@
    @@   Response message for ModelConfig.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ModelIndex:
    """
    @@
    @@  .. cpp:var:: message ModelIndex
    @@
    @@     Index entry for a model.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ModelInferRequest:
    """
    @@
    @@.. cpp:var:: message ModelInferRequest
    @@
    @@   Request message for ModelInfer.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ModelInferResponse:
    """
    @@
    @@.. cpp:var:: message ModelInferResponse
    @@
    @@   Response message for ModelInfer.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ModelMetadataRequest:
    """
    @@
    @@.. cpp:var:: message ModelMetadataRequest
    @@
    @@   Request message for ModelMetadata.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ModelMetadataResponse:
    """
    @@
    @@.. cpp:var:: message ModelMetadataResponse
    @@
    @@   Response message for ModelMetadata.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ModelReadyRequest:
    """
    @@
    @@.. cpp:var:: message ModelReadyRequest
    @@
    @@   Request message for ModelReady.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ModelReadyResponse:
    """
    @@
    @@.. cpp:var:: message ModelReadyResponse
    @@
    @@   Response message for ModelReady.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ModelRepositoryParameter:
    """
    @@
    @@.. cpp:var:: message ModelRepositoryParameter
    @@
    @@   An model repository parameter value.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ModelStatisticsRequest:
    """
    @@
    @@.. cpp:var:: message ModelStatisticsRequest
    @@
    @@   Request message for ModelStatistics.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ModelStatisticsResponse:
    """
    @@
    @@.. cpp:var:: message ModelStatisticsResponse
    @@
    @@   Response message for ModelStatistics.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ParameterChoice:
    """
    @@  .. cpp:var:: oneof parameter_choice
    @@
    @@     The parameter value can be a string, an int64,
    @@     an uint64, a double, or a boolean
    @@
    @@     Note: double and uint64 are currently
    @@           placeholders for future use and
    @@           are not supported for custom parameters
    @@
    """
    BoolParam = ParameterChoice_BoolParam
    DoubleParam = ParameterChoice_DoubleParam
    Int64Param = ParameterChoice_Int64Param
    StringParam = ParameterChoice_StringParam
    Uint64Param = ParameterChoice_Uint64Param
    @classmethod
    def bool_param(cls, value):
        ...
    @classmethod
    def double_param(cls, value):
        ...
    @classmethod
    def int64_param(cls, value):
        ...
    @classmethod
    def string_param(cls, value):
        ...
    @classmethod
    def uint64_param(cls, value):
        ...
class RegionStatus:
    """
    @@
    @@  .. cpp:var:: message RegionStatus
    @@
    @@     Status for a shared memory region.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class RepositoryIndexRequest:
    """
    @@
    @@.. cpp:var:: message RepositoryIndexRequest
    @@
    @@   Request message for RepositoryIndex.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class RepositoryIndexResponse:
    """
    @@
    @@.. cpp:var:: message RepositoryIndexResponse
    @@
    @@   Response message for RepositoryIndex.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class RepositoryModelLoadRequest:
    """
    @@
    @@.. cpp:var:: message RepositoryModelLoadRequest
    @@
    @@   Request message for RepositoryModelLoad.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class RepositoryModelLoadResponse:
    """
    @@
    @@.. cpp:var:: message RepositoryModelLoadResponse
    @@
    @@   Response message for RepositoryModelLoad.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class RepositoryModelUnloadRequest:
    """
    @@
    @@.. cpp:var:: message RepositoryModelUnloadRequest
    @@
    @@   Request message for RepositoryModelUnload.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class RepositoryModelUnloadResponse:
    """
    @@
    @@.. cpp:var:: message RepositoryModelUnloadResponse
    @@
    @@   Response message for RepositoryModelUnload.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ServerLiveResponse:
    """
    @@
    @@.. cpp:var:: message ServerLiveResponse
    @@
    @@   Response message for ServerLive.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ServerMetadataResponse:
    """
    @@
    @@.. cpp:var:: message ServerMetadataResponse
    @@
    @@   Response message for ServerMetadata.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ServerReadyResponse:
    """
    @@
    @@.. cpp:var:: message ServerReadyResponse
    @@
    @@   Response message for ServerReady.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class SystemSharedMemoryRegisterRequest:
    """
    @@
    @@.. cpp:var:: message SystemSharedMemoryRegisterRequest
    @@
    @@   Request message for SystemSharedMemoryRegister.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class SystemSharedMemoryRegisterResponse:
    """
    @@
    @@.. cpp:var:: message SystemSharedMemoryRegisterResponse
    @@
    @@   Response message for SystemSharedMemoryRegister.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class SystemSharedMemoryStatusRequest:
    """
    @@
    @@.. cpp:var:: message SystemSharedMemoryStatusRequest
    @@
    @@   Request message for SystemSharedMemoryStatus.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class SystemSharedMemoryStatusResponse:
    """
    @@
    @@.. cpp:var:: message SystemSharedMemoryStatusResponse
    @@
    @@   Response message for SystemSharedMemoryStatus.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class SystemSharedMemoryUnregisterRequest:
    """
    @@
    @@.. cpp:var:: message SystemSharedMemoryUnregisterRequest
    @@
    @@   Request message for SystemSharedMemoryUnregister.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class SystemSharedMemoryUnregisterResponse:
    """
    @@
    @@.. cpp:var:: message SystemSharedMemoryUnregisterResponse
    @@
    @@   Response message for SystemSharedMemoryUnregister.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class TensorMetadata:
    """
    @@
    @@  .. cpp:var:: message TensorMetadata
    @@
    @@     Metadata for a tensor.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def Get_shape(self):
        ...
    def Replace_shape(self, val):
        ...
    def Set_shape(self, list):
        ...
    def Take_shape(self):
        ...
class TraceSettingRequest:
    """
    @@
    @@.. cpp:var:: message TraceSettingRequest
    @@
    @@   Request message for TraceSetting.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class TraceSettingResponse:
    """
    @@
    @@.. cpp:var:: message TraceSettingResponse
    @@
    @@   Response message for TraceSetting.
    @@
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
