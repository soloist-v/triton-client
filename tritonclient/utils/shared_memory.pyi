"""Type stubs for tritonclient.utils.shared_memory module."""

from typing import Any
import numpy as np

__all__ = [
    "SharedMemoryRegion",
    "create_shared_memory_region",
    "set_shared_memory_region",
    "get_contents_as_numpy",
    "destroy_shared_memory_region",
    "mapped_shared_memory_regions",
    "SharedMemoryException",
]


class SharedMemoryRegion:
    """Handle for a system shared memory region."""
    ...


def create_shared_memory_region(
    triton_shm_name: str,
    shm_key: str,
    byte_size: int,
    create_only: bool = False,
) -> SharedMemoryRegion:
    """Create a system shared memory region.
    
    Args:
        triton_shm_name: The unique name of the shared memory region to be created.
        shm_key: The unique key of the shared memory object.
        byte_size: The size in bytes of the shared memory region to be created.
        create_only: Whether a shared memory region must be created. Default is False.
        
    Returns:
        A handle to the shared memory region (SharedMemoryRegion).
    """
    ...


def set_shared_memory_region(
    shm_handle: SharedMemoryRegion,
    input_values: list[np.ndarray] | tuple[np.ndarray, ...],
    offset: int = 0,
) -> None:
    """Set the contents of a shared memory region from numpy arrays.
    
    Args:
        shm_handle: The handle to the shared memory region.
        input_values: The list/tuple of numpy arrays to write to the shared memory region.
        offset: The offset, in bytes, into the region where you want the array copied. Default is 0.
    """
    ...


def get_contents_as_numpy(
    shm_handle: SharedMemoryRegion,
    datatype: np.dtype,
    shape: list[int] | tuple[int, ...],
    offset: int = 0,
) -> np.ndarray:
    """Get the contents of a shared memory region as a numpy array.
    
    Args:
        shm_handle: The handle to the shared memory region.
        datatype: The data type of the array.
        shape: The shape of the array.
        offset: The offset, in bytes, into the region where you want the array extracted. Default is 0.
        
    Returns:
        A numpy array view of the shared memory region.
    """
    ...


def destroy_shared_memory_region(shm_handle: SharedMemoryRegion) -> None:
    """Destroy a shared memory region.
    
    Args:
        shm_handle: The handle to the shared memory region to destroy.
    """
    ...


def mapped_shared_memory_regions() -> list[str]:
    """Get information about all mapped shared memory regions.
    
    Returns:
        The list of mapped system shared memory regions.
    """
    ...


class SharedMemoryException(Exception):
    """Exception type for shared memory related error."""
    ...

