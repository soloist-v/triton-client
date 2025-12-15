"""Type stubs for tritonclient.utils module."""

from typing import Any
import numpy as np

__all__ = [
    "triton_to_np_dtype",
]


def triton_to_np_dtype(datatype: str) -> np.dtype:
    """Convert a Triton data type string to a numpy dtype.
    
    Args:
        datatype: The Triton data type string (e.g., "FP32", "INT8", "UINT8").
        
    Returns:
        The corresponding numpy dtype.
    """
    ...

