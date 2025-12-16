"""Data type conversion utilities."""

import numpy as np
from typing import Dict

# Triton 数据类型到 NumPy 数据类型的映射
_TRITON_TO_NP_DTYPE: Dict[str, np.dtype] = {
    "BOOL": np.bool_,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
    "BYTES": np.object_,
}


def triton_to_np_dtype(datatype: str) -> np.dtype:
    """Convert a Triton data type string to a numpy dtype.
    
    Args:
        datatype: The Triton data type string (e.g., "FP32", "INT8", "UINT8").
        
    Returns:
        The corresponding numpy dtype.
        
    Raises:
        ValueError: If the datatype is not supported.
    """
    dtype = _TRITON_TO_NP_DTYPE.get(datatype.upper())
    if dtype is None:
        raise ValueError(f"Unsupported Triton data type: {datatype}")
    return dtype

