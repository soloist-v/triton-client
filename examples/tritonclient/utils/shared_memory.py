"""Shared memory utilities for Triton client."""

import multiprocessing.shared_memory as shm
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SharedMemoryException(Exception):
    """Exception type for shared memory related errors."""
    pass


class SharedMemoryRegion:
    """Handle for a system shared memory region."""
    
    def __init__(self, triton_shm_name: str, shm_key: str, byte_size: int):
        """Initialize a shared memory region handle.
        
        Args:
            triton_shm_name: The unique name of the shared memory region for Triton.
            shm_key: The unique key of the shared memory object (used by multiprocessing).
            byte_size: The size in bytes of the shared memory region.
        """
        self._triton_shm_name = triton_shm_name
        self._shm_key = shm_key
        self._byte_size = byte_size
        self._shm: Optional[shm.SharedMemory] = None
        self._created = False
    
    @property
    def name(self) -> str:
        """Get the Triton shared memory region name."""
        return self._triton_shm_name
    
    @property
    def key(self) -> str:
        """Get the shared memory key."""
        return self._shm_key
    
    @property
    def byte_size(self) -> int:
        """Get the size in bytes."""
        return self._byte_size
    
    def create(self) -> None:
        """Create a new shared memory region."""
        if self._shm is not None:
            return
        
        try:
            self._shm = shm.SharedMemory(
                create=True,
                size=self._byte_size,
                name=self._shm_key
            )
            self._created = True
            logger.debug(f"Created shared memory: {self._shm_key}, size: {self._byte_size}")
        except FileExistsError:
            raise SharedMemoryException(f"Shared memory region already exists: {self._shm_key}")
    
    def attach(self) -> None:
        """Attach to an existing shared memory region."""
        if self._shm is not None:
            return
        
        try:
            self._shm = shm.SharedMemory(name=self._shm_key)
            self._created = False
            logger.debug(f"Attached to shared memory: {self._shm_key}")
        except FileNotFoundError:
            raise SharedMemoryException(f"Shared memory region not found: {self._shm_key}")
    
    def close(self) -> None:
        """Close the shared memory region (but don't delete it)."""
        if self._shm:
            self._shm.close()
            self._shm = None
            logger.debug(f"Closed shared memory: {self._shm_key}")
    
    def unlink(self) -> None:
        """Delete the shared memory region."""
        if self._shm:
            if self._created:
                self._shm.unlink()
                logger.debug(f"Unlinked shared memory: {self._shm_key}")
            self._shm = None
    
    def get_buffer(self) -> memoryview:
        """Get a memoryview of the shared memory buffer.
        
        Returns:
            A memoryview of the shared memory buffer.
        """
        if self._shm is None:
            raise SharedMemoryException("Shared memory region not attached")
        return self._shm.buf
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


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
        
    Raises:
        SharedMemoryException: If the shared memory region cannot be created or attached.
    """
    region = SharedMemoryRegion(triton_shm_name, shm_key, byte_size)
    
    if create_only:
        region.create()
    else:
        try:
            region.attach()
        except SharedMemoryException:
            region.create()
    
    return region


def set_shared_memory_region(
    shm_handle: SharedMemoryRegion,
    input_values: List[np.ndarray] | Tuple[np.ndarray, ...],
    offset: int = 0,
) -> None:
    """Set the contents of a shared memory region from numpy arrays.
    
    Args:
        shm_handle: The handle to the shared memory region.
        input_values: The list/tuple of numpy arrays to write to the shared memory region.
        offset: The offset, in bytes, into the region where you want the array copied. Default is 0.
        
    Raises:
        SharedMemoryException: If the shared memory region is not attached or there's not enough space.
    """
    if shm_handle._shm is None:
        raise SharedMemoryException("Shared memory region not attached")
    
    current_offset = offset
    buffer = shm_handle.get_buffer()
    
    for arr in input_values:
        arr_bytes = arr.tobytes()
        arr_size = len(arr_bytes)
        
        if current_offset + arr_size > shm_handle._byte_size:
            raise SharedMemoryException(
                f"Not enough space in shared memory region. "
                f"Required: {current_offset + arr_size}, Available: {shm_handle._byte_size}"
            )
        
        buffer[current_offset:current_offset + arr_size] = arr_bytes
        current_offset += arr_size


def get_contents_as_numpy(
    shm_handle: SharedMemoryRegion,
    datatype: np.dtype,
    shape: List[int] | Tuple[int, ...],
    offset: int = 0,
) -> np.ndarray:
    """Get the contents of a shared memory region as a numpy array.
    
    Args:
        shm_handle: The handle to the shared memory region.
        datatype: The data type of the array.
        shape: The shape of the array.
        offset: The offset, in bytes, into the region where you want the array extracted. Default is 0.
        
    Returns:
        A numpy array view of the shared memory region (copy to avoid issues when shared memory is released).
        
    Raises:
        SharedMemoryException: If the shared memory region is not attached or there's not enough data.
    """
    if shm_handle._shm is None:
        raise SharedMemoryException("Shared memory region not attached")
    
    element_size = datatype.itemsize
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    total_size = total_elements * element_size
    
    if offset + total_size > shm_handle._byte_size:
        raise SharedMemoryException(
            f"Not enough data in shared memory region. "
            f"Required: {offset + total_size}, Available: {shm_handle._byte_size}"
        )
    
    buffer = shm_handle.get_buffer()
    arr = np.frombuffer(buffer[offset:offset + total_size], dtype=datatype).reshape(shape)
    
    # 返回拷贝，避免共享内存被释放后的问题
    return arr.copy()


def destroy_shared_memory_region(shm_handle: SharedMemoryRegion) -> None:
    """Destroy a shared memory region.
    
    Args:
        shm_handle: The handle to the shared memory region to destroy.
    """
    shm_handle.unlink()


def mapped_shared_memory_regions() -> List[str]:
    """Get information about all mapped shared memory regions.
    
    Note: This is a simplified implementation. The actual implementation would
    need to track all created regions, as multiprocessing.shared_memory doesn't
    provide a way to list all existing regions.
    
    Returns:
        The list of mapped system shared memory regions (currently empty as we
        don't track regions globally).
    """
    # TODO: Implement proper tracking of shared memory regions
    return []

